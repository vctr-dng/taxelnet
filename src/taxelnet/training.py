"""Group-balanced V0 training loop with MLflow tracking and model-bundle output.

L_V0 = mean over groups g [ mean over presses j in g [ mean over taxels t |mu(x_j)_t - y_jt| ] ]

The loss is applied to every real reading before averaging within its (force, Path)
group; groups are then averaged equally (or with weight n_g ** alpha, an ablation).
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from taxelnet import artifacts
from taxelnet.checkpoint import (
    ARTIFACT_CONTRACT_VERSION,
    load_checkpoint,
    save_checkpoint,
    save_model,
)
from taxelnet.config import Configs, TrainConfig
from taxelnet.dataset import GroupBatchSampler, PairedData, TaxelDataset, load_paired
from taxelnet.evaluation import (
    compute_metrics,
    predict,
    predictions_frame,
    reference_points,
)
from taxelnet.io_csv import file_sha256
from taxelnet.model import build_model
from taxelnet.plots import save_inference_plots, select_plot_samples
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import PARTITIONS, Split, make_split, split_manifest

Logger = Callable[[dict[str, float], int], object]
# called with (epoch, model, checkpoint path) after each periodic checkpoint
CheckpointHook = Callable[[int, nn.Module, Path], object]


def group_mean(
    per_press: torch.Tensor, group: torch.Tensor, alpha: float = 0.0
) -> torch.Tensor:
    """Mean within each group, then mean over groups weighted by n_g ** alpha."""
    _, inverse = torch.unique(group, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    zeros = per_press.new_zeros(n_groups)
    count = zeros.index_add(0, inverse, torch.ones_like(per_press))
    group_loss = zeros.index_add(0, inverse, per_press) / count
    weight = count**alpha
    return (weight * group_loss).sum() / weight.sum()


def group_balanced_l1(
    pred: torch.Tensor, target: torch.Tensor, group: torch.Tensor, alpha: float = 0.0
) -> torch.Tensor:
    """Mean |error| per press, mean within each group, then weighted mean over groups."""
    return group_mean((pred - target).abs().flatten(1).mean(1), group, alpha)


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_macro_mae: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_mae: float = math.inf


def train_one_epoch(
    model: nn.Module,
    ds: TaxelDataset,
    sampler: GroupBatchSampler,
    optimizer: Optimizer,
    alpha: float = 0.0,
) -> float:
    """Group-balanced loss (scaled target space), averaged over the epoch's groups."""
    model.train()
    total, n_groups = 0.0, 0
    for batch in sampler:
        idx = torch.as_tensor(batch, device=ds.x.device)
        x, y, group = ds.__getitems__(idx)
        loss = group_balanced_l1(model(x), y, group, alpha)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        k = len(torch.unique(group))
        total += loss.item() * k
        n_groups += k
    return total / n_groups


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, ds: TaxelDataset, pre: Preprocessor, alpha: float = 0.0
) -> dict[str, float]:
    """Scaled-space group-balanced loss and raw-TC micro/macro MAE of a whole partition."""
    model.eval()
    preds = torch.cat([model(ds.x[i : i + 1024]) for i in range(0, len(ds), 1024)])
    loss = group_balanced_l1(preds, ds.y, ds.group, alpha).item()
    raw_err = (pre.inverse_y(preds) - pre.inverse_y(ds.y)).abs().flatten(1).mean(1)
    if not math.isfinite(loss):
        raise FloatingPointError(f"non-finite evaluation loss {loss}")
    return {
        "loss": loss,
        "micro_mae": raw_err.mean().item(),
        "macro_mae": group_mean(raw_err, ds.group).item(),
    }


def fit(
    model: nn.Module,
    train_ds: TaxelDataset,
    val_ds: TaxelDataset,
    pre: Preprocessor,
    train_cfg: TrainConfig,
    checkpoint_dir: Path,
    logger: Logger | None = None,
    checkpoint_extra: dict | None = None,
    on_checkpoint: CheckpointHook | None = None,
) -> TrainingHistory:
    """Train; save last.pt every epoch and best.pt on the best val macro-group MAE (TC).

    Epochs are 1-based (epoch = number of completed epochs). Every `test_every` epochs,
    and at the last one, the model is also saved as checkpoints/epoch_<N>.pt and passed
    to `on_checkpoint`.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
        if train_cfg.scheduler == "cosine"
        else None
    )
    sampler = GroupBatchSampler(
        train_ds.group.cpu().numpy(), train_cfg.groups_per_batch, seed=train_cfg.seed
    )
    alpha = train_cfg.group_weight_alpha
    history = TrainingHistory()
    extra = dict(checkpoint_extra or {})

    for epoch in range(1, train_cfg.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_one_epoch(model, train_ds, sampler, optimizer, alpha)
        if scheduler is not None:
            scheduler.step()
        train_m = evaluate_loss(model, train_ds, pre, alpha)
        val_m = evaluate_loss(model, val_ds, pre, alpha)
        history.train_losses.append(train_m["loss"])
        history.val_losses.append(val_m["loss"])
        history.val_macro_mae.append(val_m["macro_mae"])

        is_best = val_m["macro_mae"] < history.best_val_macro_mae
        if is_best:
            history.best_epoch, history.best_val_macro_mae = epoch, val_m["macro_mae"]
        extra["best_val_macro_mae"] = history.best_val_macro_mae
        extra["best_epoch"] = history.best_epoch
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, extra)
        if is_best:
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, extra)

        if logger is not None:
            logger(
                {
                    **{f"train_{k}": v for k, v in train_m.items()},
                    **{f"val_{k}": v for k, v in val_m.items()},
                    "lr": lr,
                    "best_val_macro_mae": history.best_val_macro_mae,
                },
                epoch,
            )
        print(
            f"epoch {epoch:4d}  train loss {train_m['loss']:.4f}  "
            f"val loss {val_m['loss']:.4f}  val MAE micro {val_m['micro_mae']:.1f} "
            f"macro {val_m['macro_mae']:.1f} TC{'  *' if is_best else ''}",
            flush=True,
        )

        every = train_cfg.test_every
        if every > 0 and (epoch % every == 0 or epoch == train_cfg.epochs):
            path = checkpoint_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            path.parent.mkdir(exist_ok=True)
            save_checkpoint(path, model, optimizer, epoch, extra)
            if on_checkpoint is not None:
                on_checkpoint(epoch, model, path)
    return history


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    run_id: str
    metrics: dict


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _check_test_coverage(
    paired: PairedData, split: Split, test_rows: np.ndarray
) -> None:
    """The evaluated rows exactly cover the immutable test manifest."""
    manifest = json.loads(split.test_manifest_path.read_text())
    expected = {(float(f), p) for f, p in manifest["keys"]}
    got = {
        (float(paired.groups[g]["force"]), paired.groups[g]["path"])
        for g in np.unique(paired.group[test_rows])
    }
    expected_rows = sum(
        len(r["real_id"]) for r in paired.groups if (r["force"], r["path"]) in expected
    )
    if got != expected or len(test_rows) != expected_rows:
        raise AssertionError("test predictions do not cover the test manifest exactly")


def run_training(cfgs: Configs) -> RunResult:
    """Data -> split -> train-only preprocessing -> fit -> test evaluation -> bundle,
    all tracked in one MLflow run."""
    _seed_everything(cfgs.train.seed)
    data = cfgs.data
    real_csv, sim_csv = data.data_dir / data.real_csv, data.data_dir / data.sim_csv
    checksums = {
        data.real_csv: file_sha256(real_csv),
        data.sim_csv: file_sha256(sim_csv),
    }

    paired = load_paired(data)
    split = make_split(paired, data, checksums[data.real_csv])
    train_rows = split.rows["train"]
    pre = Preprocessor.fit(
        cfgs.preprocessing, paired.x[train_rows], paired.y[train_rows]
    )
    device = cfgs.train.device
    ds = {
        p: TaxelDataset(paired, split.rows[p], pre.transform_x, pre.transform_y).to(
            device
        )
        for p in PARTITIONS
    }
    for p in PARTITIONS:
        print(
            f"{p}: {len(ds[p])} readings, {len(np.unique(paired.group[split.rows[p]]))} groups"
        )

    model = build_model(cfgs.model, paired.taxel_shape).to(device)
    out = model(ds["train"].x[:1])
    if tuple(out.shape[1:]) != paired.taxel_shape:
        raise ValueError(
            f"model output {tuple(out.shape)} != taxel grid {paired.taxel_shape}"
        )
    n_params = sum(p.numel() for p in model.parameters())

    run_name = cfgs.tracking.run_name or f"{cfgs.model.name}"
    run_dir = cfgs.train.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)

    mlflow.set_tracking_uri(cfgs.tracking.resolved_uri())
    if mlflow.get_experiment_by_name(cfgs.tracking.experiment) is None:
        mlflow.create_experiment(
            cfgs.tracking.experiment, artifact_location=cfgs.tracking.artifact_location
        )
    mlflow.set_experiment(cfgs.tracking.experiment)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        git = artifacts.git_revision(Path(__file__).parent)
        mlflow.log_params(cfgs.flat())
        mlflow.log_params(
            {
                "model.n_params": n_params,
                "model.architecture": repr(model).replace("\n", " "),
                "train.optimizer": "AdamW",
                "train.loss": "group_balanced_l1",
                "train.device_name": (
                    torch.cuda.get_device_name(device)
                    if device.startswith("cuda")
                    else "cpu"
                ),
                "data.mesh_grid_shape": list(paired.mesh_shape),
                "data.taxel_grid_shape": list(paired.taxel_shape),
                "data.orientation": "tax1/n195 top-left; rows along X; no flip",
                **{f"split.{p}_readings": len(split.rows[p]) for p in PARTITIONS},
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            }
        )
        mlflow.set_tags(
            {
                "git.commit": git["commit"],
                "git.dirty": str(git["dirty"]),
                **{f"sha256.{k}": v for k, v in checksums.items()},
                **{f"version.{k}": v for k, v in artifacts.package_versions().items()},
            }
        )

        artifacts.write_json(run_dir / "config.json", cfgs.to_dict())
        artifacts.write_json(
            run_dir / "schema_report.json", artifacts.schema_report(paired, checksums)
        )
        artifacts.write_json(
            run_dir / "split_manifest.json",
            split_manifest(paired, split, data, cfgs.train.group_weight_alpha),
        )
        pre.save(run_dir / "preprocessing.json")
        artifacts.write_node_ids_csv(run_dir / "node_ids.csv")
        artifacts.write_requirements_lock(run_dir / "requirements.lock.txt")

        test_rows = split.rows["test"]
        test_target, test_group = paired.y[test_rows], paired.group[test_rows]
        plot_pos = select_plot_samples(test_target, cfgs.eval.plot_samples)
        plot_titles = [
            "{path} @ {force:g} N (row {row})".format(
                **paired.groups[paired.group[r]], row=paired.source_rows[r]
            )
            for r in test_rows[plot_pos]
        ]

        def evaluate_test(model: nn.Module, name: str, title: str) -> dict:
            """Test metrics of `model` + inference plots in plots/<name>/."""
            pred = predict(model, ds["test"], pre)
            m = compute_metrics(
                pred, test_target, test_group, cfgs.eval.ssim_data_range
            )
            out = run_dir / "plots" / name
            save_inference_plots(
                out, paired.x[test_rows], test_target, pred, m, plot_pos,
                plot_titles, title,
            )  # fmt: skip
            artifacts.write_json(out / "test_metrics.json", m)
            mlflow.log_artifacts(str(out), f"plots/{name}")
            return m

        def on_checkpoint(epoch: int, model: nn.Module, path: Path) -> None:
            # monitoring only: best.pt is selected on validation, never on test
            m = evaluate_test(model, f"epoch_{epoch:04d}", f"test, epoch {epoch}")
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in m.items() if isinstance(v, float)},
                step=epoch,
            )
            mlflow.log_artifact(str(path), "checkpoints")
            print(
                f"epoch {epoch:4d}  test MAE micro {m['micro_mae']:.1f} "
                f"macro {m['macro_mae']:.1f} TC -> plots/epoch_{epoch:04d}",
                flush=True,
            )

        history = fit(
            model,
            ds["train"],
            ds["val"],
            pre,
            cfgs.train,
            run_dir,
            logger=lambda m, step: mlflow.log_metrics(m, step=step),
            checkpoint_extra={
                "model_config": asdict(cfgs.model),
                "taxel_shape": list(paired.taxel_shape),
                "mlflow_run_id": run_id,
            },
            on_checkpoint=on_checkpoint,
        )

        load_checkpoint(run_dir / "best.pt", model, map_location=device)
        save_model(run_dir / "model.pt", model, cfgs.model, paired.taxel_shape)
        evaluate_test(model, "best", f"test, best model (epoch {history.best_epoch})")

        metrics: dict = {
            "best_epoch": history.best_epoch,
            "best_val_macro_mae": history.best_val_macro_mae,
            "units": "tactile counts (TC); MSE in TC^2",
            "note": "deterministic V0: CRPS = MAE, energy score = Euclidean error",
        }
        for p in ("val", "test"):
            rows = split.rows[p]
            pred = predict(model, ds[p], pre)
            target, group = paired.y[rows], paired.group[rows]
            metrics[p] = compute_metrics(pred, target, group, cfgs.eval.ssim_data_range)
            metrics[p]["reference"] = reference_points(target, group)
            if p == "test":
                _check_test_coverage(paired, split, rows)
                predictions_frame(paired, rows, p, pred).write_parquet(
                    run_dir / "predictions_test.parquet"
                )
        metrics["integrity"] = {
            "scalers_fitted_on": pre.fitted_on,
            "scaler_fit_rows": pre.n_fit_rows,
            "train_rows": len(train_rows),
            "test_covers_manifest": True,
            "partitions_disjoint": True,
        }
        assert pre.n_fit_rows == len(train_rows)
        artifacts.write_json(run_dir / "metrics.json", metrics)

        mlflow.log_metrics(
            {
                "best_epoch": history.best_epoch,
                # per-epoch series are train_* / val_* / test_* (every test_every
                # epochs); the best-model evaluation is final_val_* / final_test_*
                **{
                    f"final_{p}_{k}": v
                    for p in ("val", "test")
                    for k, v in metrics[p].items()
                    if isinstance(v, float)
                },
                **{
                    f"final_test_ref_{ref}_{k}": v
                    for ref, d in metrics["test"]["reference"].items()
                    for k, v in d.items()
                    if isinstance(v, float)
                },
                **{
                    f"final_test_taxel{t + 1}_mae": v
                    for t, v in enumerate(metrics["test"]["per_taxel_micro_mae"])
                },
            }
        )
        # plots/ and checkpoints/ were uploaded as they were made
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                mlflow.log_artifact(str(path))

    return RunResult(run_dir, run_id, metrics)
