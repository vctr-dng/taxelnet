"""Group-balanced V0 training loop with MLflow tracking and model-bundle output.

L_V0 = mean over groups g [ mean over presses j in g [ mean over taxels t |mu(x_j)_t - y_jt| ] ]

The loss is applied to every real reading before averaging within its (force, Path)
group; groups are then averaged equally (or with weight n_g ** alpha, an ablation).
"""

from __future__ import annotations

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

from taxelnet import amp, artifacts
from taxelnet.checkpoint import (
    ARTIFACT_CONTRACT_VERSION,
    load_checkpoint,
    save_checkpoint,
    save_model,
)
from taxelnet.config import Configs, TrainConfig
from taxelnet.dataset import (
    GroupBatchSampler,
    PairedData,
    TaxelDataset,
    load_paired,
    load_test_paired,
)
from taxelnet.evaluation import (
    compute_metrics,
    global_ssim,
    object_metrics,
    predict,
    predictions_frame,
    reference_points,
)
from taxelnet.io_csv import file_sha256
from taxelnet.model import build_model
from taxelnet.plots import (
    plot_group_label,
    save_inference_plots,
    select_plot_samples,
)
from taxelnet.preprocessing import Preprocessor
from taxelnet.ssi import ssi_losses
from taxelnet.saturation import head_bounds
from taxelnet.splits import PARTITIONS, TEST, make_split, shared_keys, split_manifest

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
    pre: Preprocessor | None = None,
    *,
    residual_l1_weight: float = 0.0,
    ssi_weight: float = 0.0,
    grad_weight: float = 0.0,
    ssi_eps: float = 0.05,
    logamp_weight: float = 0.0,
    logamp_eps: float = 50.0,
) -> dict[str, float]:
    """Group-balanced loss (scaled target space), averaged over the epoch's groups.

    Optional terms: `residual_l1_weight` times the mean |model.residual| (hybrid models
    only); the ssi / grad pattern terms (taxelnet.ssi); the group-balanced log-amplitude
    term computed in raw TC (`taxelnet.amp.logamp_per_press`, needs `pre`). Returns the
    epoch mean of each term: l1, residual / ssi / grad / logamp (when enabled), total.
    """
    model.train()
    totals: dict[str, float] = {}
    n_groups = 0
    for batch in sampler:
        idx = torch.as_tensor(batch, device=ds.x.device)
        x, y, group = ds.__getitems__(idx)
        pred = model(x)
        terms = {"l1": group_balanced_l1(pred, y, group, alpha)}
        loss = terms["l1"]
        if residual_l1_weight > 0:
            terms["residual"] = model.residual.abs().mean()
            loss = loss + residual_l1_weight * terms["residual"]
        if ssi_weight > 0 or grad_weight > 0:
            l_ssi, l_grad = ssi_losses(pred, y, ssi_eps)
            terms["ssi"] = group_mean(l_ssi, group, alpha)
            terms["grad"] = group_mean(l_grad, group, alpha)
            loss = loss + ssi_weight * terms["ssi"] + grad_weight * terms["grad"]
        if logamp_weight > 0:
            assert pre is not None
            per_press = amp.logamp_per_press(
                pre.inverse_y(pred, clamp=False), pre.inverse_y(y), logamp_eps
            )
            terms["logamp"] = group_mean(per_press, group, alpha)
            loss = loss + logamp_weight * terms["logamp"]
        terms["total"] = loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        k = len(torch.unique(group))
        for name, v in terms.items():
            totals[name] = totals.get(name, 0.0) + v.item() * k
        n_groups += k
    return {name: v / n_groups for name, v in totals.items()}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, ds: TaxelDataset, pre: Preprocessor, alpha: float = 0.0
) -> dict[str, float]:
    """Scaled-space group-balanced loss and raw-TC micro/macro MAE of a whole partition."""
    model.eval()
    preds = torch.cat([model(ds.x[i : i + 1024]) for i in range(0, len(ds), 1024)])
    loss = group_balanced_l1(preds, ds.y, ds.group, alpha).item()
    raw_err = (
        (pre.inverse_y(preds, group=ds.sensor) - pre.inverse_y(ds.y, group=ds.sensor))
        .abs().flatten(1).mean(1)
    )  # fmt: skip
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
    params: list | object = model.parameters()
    if train_cfg.physics_lr is not None and hasattr(model, "footprint"):
        phys = {id(p) for p in model.footprint.parameters()}
        params = [
            {"params": [p for p in model.parameters() if id(p) not in phys]},
            {"params": list(model.footprint.parameters()), "lr": train_cfg.physics_lr},
        ]
    optimizer = torch.optim.AdamW(
        params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
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
        epoch_terms = train_one_epoch(
            model, train_ds, sampler, optimizer, alpha, pre,
            residual_l1_weight=train_cfg.residual_l1_weight,
            ssi_weight=train_cfg.loss_ssi_weight,
            grad_weight=train_cfg.loss_grad_weight,
            ssi_eps=train_cfg.loss_ssi_eps,
            logamp_weight=train_cfg.loss_logamp_weight,
            logamp_eps=train_cfg.loss_logamp_eps,
        )  # fmt: skip
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
                    # running epoch means of the optimized terms (train_l1, ...)
                    **{f"train_{k}": v for k, v in epoch_terms.items()},
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


def run_training(cfgs: Configs) -> RunResult:
    """Training data -> train/val split -> train-only preprocessing -> fit -> evaluation
    on the external test pair -> bundle, all tracked in one MLflow run."""
    _seed_everything(cfgs.train.seed)
    data = cfgs.data

    paired = load_paired(data)
    test = load_test_paired(data)
    if (test.mesh_shape, test.taxel_shape) != (paired.mesh_shape, paired.taxel_shape):
        raise ValueError(
            f"test grids {test.mesh_shape} -> {test.taxel_shape} differ from training "
            f"grids {paired.mesh_shape} -> {paired.taxel_shape}"
        )
    shared = shared_keys(paired, test)
    if shared:
        raise ValueError(
            f"{len(shared)} test keys also occur in the training data, e.g. {shared[:5]}"
        )
    checksums = {f.name: file_sha256(f) for f in (*paired.files, *test.files)}

    split = make_split(paired, data)
    if split.held_out_shapes:
        print(f"held-out val shapes: {', '.join(split.held_out_shapes)}")
    train_rows = split.rows["train"]
    pre = Preprocessor.fit(
        cfgs.preprocessing,
        paired.x[train_rows],
        paired.y[train_rows],
        None if paired.sensor is None else paired.sensor[train_rows],
    )
    if cfgs.preprocessing.target_group_key != "none":
        if cfgs.train.loss_logamp_weight > 0:
            raise ValueError("loss_logamp_weight is not supported with target_group_key")
        print(f"per-group target scalers: {pre.group_labels}")
    device = cfgs.train.device
    # partition -> (paired data, rows): train/val from the training pair, test external
    sources: dict[str, tuple[PairedData, np.ndarray]] = {
        **{p: (paired, split.rows[p]) for p in PARTITIONS},
        TEST: (test, np.arange(len(test.group))),
    }
    ds = {
        p: TaxelDataset(src, rows, pre.transform_x, pre.transform_y).to(device)
        for p, (src, rows) in sources.items()
    }
    for p, (src, rows) in sources.items():
        print(f"{p}: {len(ds[p])} readings, {len(np.unique(src.group[rows]))} groups")

    if cfgs.model.head_activation == "tanh_bounded":
        lo, hi = head_bounds(
            pre.transform_y(
                paired.y[train_rows],
                None if paired.sensor is None else paired.sensor[train_rows],
            ),
            cfgs.model.head_bound_margin,
        )
        if cfgs.model.head_lower is None:
            cfgs.model.head_lower = lo
        if cfgs.model.head_upper is None:
            cfgs.model.head_upper = hi
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
        try:
            mlflow.create_experiment(
                cfgs.tracking.experiment,
                artifact_location=cfgs.tracking.artifact_location,
            )
        except mlflow.exceptions.MlflowException:
            # a concurrent run created it first
            if mlflow.get_experiment_by_name(cfgs.tracking.experiment) is None:
                raise
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
                "train.loss": "group_balanced_l1"
                + ("+residual_l1" if cfgs.train.residual_l1_weight > 0 else "")
                + ("+ssi" if cfgs.train.loss_ssi_weight > 0 else "")
                + ("+grad" if cfgs.train.loss_grad_weight > 0 else "")
                + ("+logamp" if cfgs.train.loss_logamp_weight > 0 else ""),
                "train.device_name": (
                    torch.cuda.get_device_name(device)
                    if device.startswith("cuda")
                    else "cpu"
                ),
                "data.mesh_grid_shape": list(paired.mesh_shape),
                "data.taxel_grid_shape": list(paired.taxel_shape),
                "data.orientation": "tax1/n195 top-left; rows along X; no flip",
                **{
                    f"split.{p}_readings": len(rows) for p, (_, rows) in sources.items()
                },
                "split.held_out_shapes": ",".join(split.held_out_shapes) or "none",
                "split.n_held_out_shapes": len(split.held_out_shapes),
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
            run_dir / "schema_report.json",
            {
                "train": artifacts.schema_report(paired, checksums),
                TEST: artifacts.schema_report(test, checksums, exclusions=None),
            },
        )
        artifacts.write_json(
            run_dir / "split_manifest.json",
            split_manifest(
                paired, split, data, test, checksums, cfgs.train.group_weight_alpha
            ),
        )
        pre.save(run_dir / "preprocessing.json")
        artifacts.write_node_ids_csv(run_dir / "node_ids.csv")
        artifacts.write_requirements_lock(run_dir / "requirements.lock.txt")

        test_target, test_group = test.y, test.group
        test_names = [test.groups[g]["path"] for g in test_group]

        def reading_titles(src: PairedData, rows: np.ndarray) -> list[str]:
            return [
                "{path} @ {force:g} (row {row})".format(
                    **src.groups[src.group[r]], row=src.source_rows[r]
                )
                for r in rows
            ]

        def group_labels(src: PairedData, rows: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    plot_group_label(src.groups[g]["path"], src.groups[g]["force"])
                    for g in src.group[rows]
                ]
            )

        plot_pos = select_plot_samples(test_target, cfgs.eval.plot_samples)
        plot_titles = reading_titles(test, plot_pos)
        test_rows = np.arange(len(test_group))
        test_all_titles = reading_titles(test, test_rows)
        test_labels = group_labels(test, test_rows)

        def evaluate_test(model: nn.Module, name: str, title: str) -> dict:
            """Test metrics of `model` + inference plots in plots/<name>/."""
            pred = predict(model, ds["test"], pre)
            m = compute_metrics(
                pred, test_target, test_group, cfgs.eval.ssim_data_range
            )
            m |= object_metrics(pred, test_target, test_group, test_names)
            out = run_dir / "plots" / name
            save_inference_plots(
                out, test.x, test_target, pred, m, plot_pos,
                plot_titles, title, group_labels=test_labels,
                ssim=global_ssim(pred, test_target, cfgs.eval.ssim_data_range),
                titles=test_all_titles, per_group=cfgs.eval.plot_samples_per_group,
            )  # fmt: skip
            artifacts.write_json(out / "test_metrics.json", m)
            mlflow.log_artifacts(str(out), f"plots/{name}")
            return m

        def on_checkpoint(epoch: int, model: nn.Module, path: Path) -> None:
            # monitoring only: best.pt is selected on validation, never on test
            m = evaluate_test(model, f"epoch_{epoch:04d}", f"test, epoch {epoch}")
            # decomposed val metrics at the same epochs (val_micro/macro_mae are
            # already logged every epoch by fit)
            vsrc, vrows = sources["val"]
            vm = compute_metrics(
                predict(model, ds["val"], pre),
                vsrc.y[vrows],
                vsrc.group[vrows],
                cfgs.eval.ssim_data_range,
            )
            mlflow.log_metrics(
                {
                    **{f"test_{k}": v for k, v in m.items() if isinstance(v, float)},
                    **{
                        f"val_{k}": v
                        for k, v in vm.items()
                        if isinstance(v, float) and k not in ("micro_mae", "macro_mae")
                    },
                },
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
        if hasattr(model, "report"):  # physics models: learned params + footprint plot
            phys = model.report(run_dir)
            mlflow.log_metrics({f"final_phys_{k}": v for k, v in phys.items()})
        evaluate_test(model, "best", f"test, best model (epoch {history.best_epoch})")
        # same per-group plots for the validation partition, best model only
        vsrc, vrows = sources["val"]
        if len(vrows):
            vpred, vtarget = predict(model, ds["val"], pre), vsrc.y[vrows]
            vm = compute_metrics(
                vpred, vtarget, vsrc.group[vrows], cfgs.eval.ssim_data_range
            )
            vtitles = reading_titles(vsrc, vrows)
            vpos = select_plot_samples(vtarget, cfgs.eval.plot_samples)
            out = run_dir / "plots" / "best_val"
            save_inference_plots(
                out, vsrc.x[vrows], vtarget, vpred, vm, vpos,
                [vtitles[p] for p in vpos],
                f"val, best model (epoch {history.best_epoch})",
                group_labels=group_labels(vsrc, vrows),
                ssim=global_ssim(vpred, vtarget, cfgs.eval.ssim_data_range),
                titles=vtitles, per_group=cfgs.eval.plot_samples_per_group,
            )  # fmt: skip
            mlflow.log_artifacts(str(out), "plots/best_val")

        metrics: dict = {
            "best_epoch": history.best_epoch,
            "best_val_macro_mae": history.best_val_macro_mae,
            "units": "tactile counts (TC); MSE in TC^2",
            "note": "deterministic V0: CRPS = MAE, energy score = Euclidean error",
        }
        for p in ("val", TEST):
            src, rows = sources[p]
            pred = predict(model, ds[p], pre)
            target, group = src.y[rows], src.group[rows]
            metrics[p] = compute_metrics(pred, target, group, cfgs.eval.ssim_data_range)
            metrics[p]["reference"] = reference_points(target, group)
            if p == TEST:
                obj = object_metrics(pred, target, group, test_names)
                metrics[p]["reference"]["trivial_zero"] |= obj.pop("reference")
                metrics[p] |= obj
                if len(pred) != len(test.group):
                    raise AssertionError("test predictions do not cover the test file")
                predictions_frame(src, rows, p, pred).write_parquet(
                    run_dir / "predictions_test.parquet"
                )
        metrics["integrity"] = {
            "scalers_fitted_on": pre.fitted_on,
            "scaler_fit_rows": pre.n_fit_rows,
            "train_rows": len(train_rows),
            "test_predictions_cover_test_file": True,
            "train_val_disjoint": True,
            "test_keys_disjoint_from_training": True,
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
