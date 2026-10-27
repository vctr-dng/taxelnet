"""Multi-objective Optuna search over the standard CNN (cnn_v0): SSIM, Pearson, MAE, cosine.

Simulated deformation (77 x 49) in, real taxels (7 x 4) out. Every trial trains on the
whole training pair (minus the blacklist) and is scored on the validation pair:

- SSIM: mean over readings of `windowed_ssim` (`hpo.ssim_window`, 2 x 2 by default)
- Pearson, cosine: mean over readings of the per-map correlation / cosine
- MAE: mean absolute error over readings and taxels, in TC

The validation pads are not the training pads, so with `hpo.calibrate` the predictions
get one L1 gain per pad label (S1 / S2), fitted on all readings of that pad, before MAE
and SSIM (Pearson and cosine do not depend on a positive gain). Raw MAE / SSIM are kept
in the trial's user attributes.

The scored model is the last epoch (end of the cosine schedule): `fit` keeps best.pt on
the raw val MAE alone, which would favour one objective over the others.

Every `hpo.plot_every` epochs (and at the last one) each seed's model is evaluated on the
whole val pair and on each (object, force, pad) group: objectives logged to MLflow as
val_<group>_<objective>, plots in seed<k>/plots/epoch_<N>/. The evaluation with the best
calibrated val MAE is kept (best_cal.pt) and re-plotted in seed<k>/plots/best/, with the
learning curves and eval_history.csv. With several seeds, trial_<n>/plots/ holds the same
objectives by group as violins over the seeds.
"""

from __future__ import annotations

import copy
import gc
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import polars as pl
import torch
from matplotlib.figure import Figure

from taxelnet import artifacts
from taxelnet.calibration import fit_global_gain
from taxelnet.checkpoint import load_checkpoint, load_model, save_model
from taxelnet.config import HPOConfigs
from taxelnet.dataset import (
    PairedData,
    TaxelDataset,
    drop_names,
    load_paired,
    load_paired_files,
    load_test_paired,
    read_blacklist,
)
from taxelnet.evaluation import (
    compute_metrics,
    cosine_similarity,
    global_ssim,
    pearson,
    predict,
    windowed_ssim,
)
from taxelnet.model import build_model
from taxelnet.plots import (
    plot_learning_curves,
    plot_metrics_by_group,
    plot_per_taxel_mae,
    plot_scatter,
    reading_group_labels,
    reading_titles,
    save_figures,
    save_inference_plots,
    select_plot_samples,
)
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import shared_keys
from taxelnet.training import _seed_everything, fit, setup_mlflow

# TPESampler(group=True) and Study.set_metric_names are used on purpose
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

OBJECTIVES = ("ssim", "pearson", "mae", "cosine")
DIRECTIONS = ("maximize", "maximize", "minimize", "maximize")


@dataclass(frozen=True)
class HPOData:
    train: PairedData
    val: PairedData
    test: PairedData
    n_blacklisted: dict[str, int]


def load_hpo_data(cfgs: HPOConfigs) -> HPOData:
    """Training, validation and test pairs, blacklist removed (strict on training)."""
    hpo, data = cfgs.hpo, cfgs.data
    train = load_paired(data)
    val = load_paired_files(
        hpo.val_dir / hpo.val_real_csv,
        hpo.val_dir / hpo.val_sim_csv,
        list(hpo.val_key_cols),
        data.cache_dir,
    )
    test = load_test_paired(data)
    dropped = {"train": 0, "val": 0, "test": 0}
    if hpo.blacklist_csv is not None:
        names = read_blacklist(hpo.blacklist_csv)
        train, dropped["train"] = drop_names(train, names, strict=True)
        val, dropped["val"] = drop_names(val, names, strict=False)
        test, dropped["test"] = drop_names(test, names, strict=False)
    for name, pair in (("val", val), ("test", test)):
        if (pair.mesh_shape, pair.taxel_shape) != (train.mesh_shape, train.taxel_shape):
            raise ValueError(
                f"{name} grids {pair.mesh_shape} -> {pair.taxel_shape} differ from "
                f"training grids {train.mesh_shape} -> {train.taxel_shape}"
            )
    shared = shared_keys(train, val)
    if shared:
        raise ValueError(
            f"{len(shared)} val keys also occur in the training data, e.g. {shared[:5]}"
        )
    return HPOData(train, val, test, dropped)


def calibrate_pads(
    pred: np.ndarray,
    target: np.ndarray,
    pads: np.ndarray | None,
    calibrate: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Predictions (float64) with each pad's exact L1 gain applied, and the gains.

    `pads` (N,) is each reading's pad label; None = one pad. Without `calibrate`, the
    predictions are returned unchanged and there are no gains.
    """
    labels = np.full(len(pred), "all") if pads is None else np.asarray(pads).astype(str)
    cal = pred.astype(np.float64)
    gains: dict[str, float] = {}
    if calibrate:
        cal = cal.copy()
        for q in sorted(set(labels)):
            sel = labels == q
            gain = float(fit_global_gain(pred[sel], target[sel]).gain)
            cal[sel] *= gain
            gains[f"gain_{q}"] = gain
    return cal, gains


def reading_objectives(
    cal: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float,
    window: tuple[int, int] = (2, 2),
) -> dict[str, np.ndarray]:
    """Per-reading objectives (N,) of the calibrated `cal`, plus raw SSIM / MAE of `pred`."""
    n = len(pred)
    return {
        "ssim": windowed_ssim(cal, target, data_range, window),
        "pearson": pearson(cal, target),
        "mae": np.abs(cal - target).reshape(n, -1).mean(1),
        "cosine": cosine_similarity(cal, target),
        "raw_ssim": windowed_ssim(pred, target, data_range, window),
        "raw_mae": np.abs(pred.astype(np.float64) - target).reshape(n, -1).mean(1),
    }


def objective_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pads: np.ndarray | None,
    data_range: float,
    window: tuple[int, int] = (2, 2),
    calibrate: bool = True,
) -> dict[str, float]:
    """The four objectives (raw TC), plus raw MAE / SSIM and the fitted pad gains.

    `pads` (N,) is each reading's pad label; None = one pad. With `calibrate`, each pad's
    predictions are multiplied by the exact L1 gain fitted on all of its readings.
    """
    cal, gains = calibrate_pads(pred, target, pads, calibrate)
    per_reading = reading_objectives(cal, pred, target, data_range, window)
    return gains | {k: float(v.mean()) for k, v in per_reading.items()}


def group_objectives(
    per_reading: dict[str, np.ndarray], labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """{"all": mean of each per-reading objective, <label>: mean over its readings}."""
    labels = np.asarray(labels).astype(str)
    out = {"all": {k: float(v.mean()) for k, v in per_reading.items()}}
    for g in sorted(set(labels)):
        sel = labels == g
        out[g] = {k: float(v[sel].mean()) for k, v in per_reading.items()}
    return out


def suggest_configs(trial: optuna.trial.BaseTrial, base: HPOConfigs) -> HPOConfigs:
    """A copy of `base` with the trial's standard-CNN hyperparameters."""
    cfgs = copy.deepcopy(base)
    m, t, p = cfgs.model, cfgs.train, cfgs.preprocessing
    m.name = "cnn_v0"  # standard CNN: no CoordConv, no physics hybrid
    m.base_channels = trial.suggest_categorical("base_channels", [8, 16, 32, 64])
    m.num_down_blocks = trial.suggest_int("num_down_blocks", 1, 4)
    m.taxel_convs = trial.suggest_int("taxel_convs", 0, 2)
    m.taxel_channels = (
        trial.suggest_categorical("taxel_channels", [32, 64, 128])
        if m.taxel_convs > 0
        else 0
    )
    m.taxel_affine_head = trial.suggest_categorical("taxel_affine_head", [False, True])
    t.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    t.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    t.groups_per_batch = trial.suggest_categorical(
        "groups_per_batch", [16, 32, 64, 128, 256]
    )
    t.loss_ssi_weight = trial.suggest_categorical(
        "loss_ssi_weight", [0.0, 0.1, 0.3, 1.0]
    )
    # alpha * L1 + (1 - alpha) * (1 - SSIM), SSIM window = the objective's
    t.loss_l1_ssim_alpha = trial.suggest_float("loss_l1_ssim_alpha", 0.0, 1.0)
    t.loss_ssim_window = cfgs.hpo.ssim_window
    t.loss_ssim_data_range = cfgs.eval.ssim_data_range
    t.scheduler = "cosine"
    p.target_group_key = trial.suggest_categorical(
        "target_group_key", ["none", "sensor"]
    )
    p.target_scaler, p.input_scaler = "global_scale", "max_abs"
    return cfgs


def _study_dir(cfgs: HPOConfigs) -> Path:
    return cfgs.hpo.output_dir / cfgs.hpo.study_name


ByGroup = dict[str, dict[str, float]]  # {"all" | group: {objective: value}}


def _units(cfgs: HPOConfigs) -> dict[str, str]:
    return {"mae": "TC, calibrated" if cfgs.hpo.calibrate else "TC"}


def _seed_values(by_groups: list[ByGroup]) -> dict[str, dict[str, list[float]]]:
    """{group: {objective: one value per seed}} from each seed's `ByGroup`."""
    return {
        g: {k: [b[g][k] for b in by_groups] for k in OBJECTIVES} for g in by_groups[0]
    }


@dataclass
class SeedRecord:
    """Periodic val evaluations of one seed: epoch -> objectives by group, and the best."""

    history: dict[int, ByGroup] = field(default_factory=dict)
    best_epoch: int = -1
    best: ByGroup = field(default_factory=dict)


class ValMonitor:
    """`fit` checkpoint hook of one seed: calibrated val objectives on the whole val pair
    and on each group, logged and plotted in plots/epoch_<N>/. Keeps the periodic
    checkpoint of the best calibrated MAE and deletes the others."""

    def __init__(
        self,
        cfgs: HPOConfigs,
        val: PairedData,
        val_ds: TaxelDataset,
        pre: Preprocessor,
        out_dir: Path,
        prefix: str,
    ) -> None:
        self.cfgs, self.val, self.val_ds, self.pre = cfgs, val, val_ds, pre
        # MLflow: metrics <prefix>val_<group>_<objective>, artifacts under <prefix>/
        self.out_dir, self.prefix = out_dir, prefix
        self.artifact_dir = prefix.rstrip("_")
        rows = np.arange(len(val.group))
        self.labels = reading_group_labels(val, rows)
        self.titles = reading_titles(val, rows)
        self.record = SeedRecord()
        self.best_mae = math.inf
        self.best_path: Path | None = None

    def _artifact_path(self, *parts: str) -> str:
        return "/".join([self.artifact_dir, *parts] if self.artifact_dir else parts)

    def evaluate(self, model: torch.nn.Module) -> tuple[np.ndarray, ByGroup, dict]:
        """Calibrated predictions, objectives by group, pad gains."""
        hpo, val = self.cfgs.hpo, self.val
        pred = predict(model, self.val_ds, self.pre)
        cal, gains = calibrate_pads(pred, val.y, val.sensor, hpo.calibrate)
        per_reading = reading_objectives(
            cal, pred, val.y, self.cfgs.eval.ssim_data_range, hpo.ssim_window
        )
        return cal, group_objectives(per_reading, self.labels), gains

    def write(
        self, name: str, title: str, cal: np.ndarray, by_group: ByGroup, info: dict
    ) -> None:
        """plots/<name>/: objectives by group, inference plots on the whole val pair and
        scatter / per-taxel MAE of each group, metrics.json; logged to MLflow."""
        val, data_range = self.val, self.cfgs.eval.ssim_data_range
        out = self.out_dir / "plots" / name
        if self.cfgs.hpo.calibrate:
            title += " (calibrated per pad)"
        figures = {
            "metrics_by_group.png": plot_metrics_by_group(
                _seed_values([by_group]), OBJECTIVES, _units(self.cfgs)
            )
        }
        for g in sorted(set(self.labels)):
            sel = np.flatnonzero(self.labels == g)
            m = compute_metrics(cal[sel], val.y[sel], val.group[sel], data_range)
            figures[f"groups/{g}/scatter.png"] = plot_scatter(cal[sel], val.y[sel], m)
            figures[f"groups/{g}/per_taxel_mae.png"] = plot_per_taxel_mae(
                m["per_taxel_micro_mae"], val.taxel_shape
            )
        save_figures(out, figures, title)
        pos = select_plot_samples(val.y, self.cfgs.eval.plot_samples)
        save_inference_plots(
            out, val.x, val.y, cal, compute_metrics(cal, val.y, val.group, data_range),
            pos, [self.titles[p] for p in pos], title, group_labels=self.labels,
            ssim=global_ssim(cal, val.y, data_range), titles=self.titles,
            per_group=self.cfgs.eval.plot_samples_per_group,
        )  # fmt: skip
        artifacts.write_json(out / "metrics.json", {**info, "groups": by_group})
        mlflow.log_artifacts(str(out), self._artifact_path("plots", name))

    def __call__(self, epoch: int, model: torch.nn.Module, path: Path) -> None:
        cal, by_group, gains = self.evaluate(model)
        self.record.history[epoch] = by_group
        self.write(
            f"epoch_{epoch:04d}", f"val, epoch {epoch}", cal, by_group,
            {"epoch": epoch, **gains},
        )  # fmt: skip
        mlflow.log_metrics(
            {
                f"{self.prefix}val_{g}_{k}": v
                for g, d in by_group.items()
                for k, v in d.items()
            },
            step=epoch,
        )
        if by_group["all"]["mae"] < self.best_mae:
            if self.best_path is not None:
                self.best_path.unlink()
            self.best_mae, self.best_path = by_group["all"]["mae"], path
            self.record.best_epoch = epoch
        else:
            path.unlink()

    def finish(self, model: torch.nn.Module, last_epoch: int) -> SeedRecord:
        """Load the best periodic checkpoint into `model` (kept as best_cal.pt), plot it
        in plots/best/; eval_history.csv and plots/learning_curves.png. Without periodic
        evaluations, `model` (the last epoch) is the best."""
        if self.best_path is not None:
            load_checkpoint(self.best_path, model, map_location=self.cfgs.train.device)
            self.best_path.replace(self.out_dir / "best_cal.pt")
            if not any(self.best_path.parent.iterdir()):
                self.best_path.parent.rmdir()
        else:
            self.record.best_epoch = last_epoch
        cal, self.record.best, gains = self.evaluate(model)
        epoch = self.record.best_epoch
        self.write(
            "best", f"val, best epoch {epoch} by calibrated MAE", cal,
            self.record.best, {"epoch": epoch, **gains},
        )  # fmt: skip
        history = self.record.history
        if history:
            epochs = sorted(history)
            pl.DataFrame(
                [
                    {"epoch": e, "group": g, **d}
                    for e in epochs
                    for g, d in history[e].items()
                ]
            ).write_csv(self.out_dir / "eval_history.csv")
            curves = {
                g: {k: [history[e][g][k] for e in epochs] for k in OBJECTIVES}
                for g in history[epochs[0]]
            }
            out = self.out_dir / "plots"
            save_figures(
                out,
                {
                    "learning_curves.png": plot_learning_curves(
                        epochs, curves, OBJECTIVES, epoch
                    )
                },
                "val objectives by group",
            )
            mlflow.log_artifact(
                str(out / "learning_curves.png"), self._artifact_path("plots")
            )
            mlflow.log_artifact(
                str(self.out_dir / "eval_history.csv"), self.artifact_dir or None
            )
        return self.record


def _train_seed(
    cfgs: HPOConfigs, data: HPOData, seed: int, out_dir: Path, log_prefix: str
) -> tuple[dict[str, Any], SeedRecord]:
    """Train one model on the whole training pair; save its bundle; val objectives of the
    last epoch, and the periodic evaluations by group."""
    _seed_everything(seed)
    train_cfg = copy.deepcopy(cfgs.train)
    train_cfg.seed, train_cfg.test_every = seed, cfgs.hpo.plot_every
    device = train_cfg.device
    pre = Preprocessor.fit(
        cfgs.preprocessing, data.train.x, data.train.y, data.train.sensor
    )
    train_ds = TaxelDataset(
        data.train, np.arange(len(data.train.group)), pre.transform_x, pre.transform_y
    ).to(device)
    val_ds = TaxelDataset(
        data.val, np.arange(len(data.val.group)), pre.transform_x, pre.transform_y
    ).to(device)
    model = build_model(cfgs.model, data.train.taxel_shape).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    monitor = ValMonitor(cfgs, data.val, val_ds, pre, out_dir, log_prefix)
    try:
        fit(
            model,
            train_ds,
            val_ds,
            pre,
            train_cfg,
            out_dir,
            logger=lambda m, step: mlflow.log_metrics(
                {f"{log_prefix}{k}": v for k, v in m.items()}, step=step
            ),
            on_checkpoint=monitor,
        )
        pred = predict(model, val_ds, pre)
        metrics = objective_metrics(
            pred,
            data.val.y,
            data.val.sensor,
            cfgs.eval.ssim_data_range,
            cfgs.hpo.ssim_window,
            cfgs.hpo.calibrate,
        )
        save_model(out_dir / "model.pt", model, cfgs.model, data.train.taxel_shape)
        pre.save(out_dir / "preprocessing.json")
        artifacts.write_json(out_dir / "val_metrics.json", metrics)
        record = monitor.finish(model, train_cfg.epochs)  # loads the best weights
    finally:
        del model, train_ds, val_ds, monitor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {**metrics, "n_params": n_params}, record


def _write_trial_plots(
    cfgs: HPOConfigs, trial_dir: Path, records: list[SeedRecord]
) -> None:
    """trial_<n>/plots/: objectives by group as violins over the seeds at each periodic
    epoch and at each seed's best epoch; learning curves (seed mean, min-max band)."""
    n, units = len(records), _units(cfgs)
    out = trial_dir / "plots"
    epochs = sorted(set.intersection(*(set(r.history) for r in records)))
    for e in epochs:
        save_figures(
            out / f"epoch_{e:04d}",
            {
                "metrics_by_group.png": plot_metrics_by_group(
                    _seed_values([r.history[e] for r in records]), OBJECTIVES, units
                )
            },
            f"val over {n} seeds, epoch {e}",
        )
    best_epochs = ", ".join(str(r.best_epoch) for r in records)
    figures = {
        "best/metrics_by_group.png": plot_metrics_by_group(
            _seed_values([r.best for r in records]), OBJECTIVES, units
        )
    }
    if epochs:
        per_epoch = [_seed_values([r.history[e] for r in records]) for e in epochs]
        stat = {
            name: {
                g: {k: [float(fn(v[g][k])) for v in per_epoch] for k in OBJECTIVES}
                for g in per_epoch[0]
            }
            for name, fn in (("mean", np.mean), ("min", np.min), ("max", np.max))
        }
        bands = {
            g: {k: (stat["min"][g][k], stat["max"][g][k]) for k in OBJECTIVES}
            for g in per_epoch[0]
        }
        figures["learning_curves.png"] = plot_learning_curves(
            epochs, stat["mean"], OBJECTIVES, bands=bands
        )
    save_figures(out, figures, f"val over {n} seeds (best epochs {best_epochs})")
    mlflow.log_artifacts(str(out), "plots")


def run_trial(
    trial: optuna.Trial, data: HPOData, base: HPOConfigs
) -> tuple[float, ...]:
    """Train `seeds_per_trial` models of the trial's config; mean val objectives."""
    cfgs = suggest_configs(trial, base)
    trial_dir = _study_dir(cfgs) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifacts.write_json(trial_dir / "config.json", cfgs.to_dict())
    n_seeds = cfgs.hpo.seeds_per_trial
    seeds = [cfgs.train.seed + k for k in range(n_seeds)]

    run_name = f"{cfgs.hpo.study_name}-t{trial.number:04d}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            cfgs.flat()
            | {f"trial.{k}": v for k, v in trial.params.items()}
            | {
                "optuna.study": cfgs.hpo.study_name,
                "optuna.trial_number": trial.number,
                # unique over every study of the storage
                "optuna.trial_id": trial._trial_id,
            }
        )
        mlflow.set_tags(
            {
                "optuna.study": cfgs.hpo.study_name,
                "optuna.trial": str(trial.number),
                "git.commit": artifacts.git_revision(Path(__file__).parent)["commit"],
            }
        )
        trial.set_user_attr("mlflow_run_id", run.info.run_id)
        per_seed, records = [], []
        for seed in seeds:
            metrics, record = _train_seed(
                cfgs,
                data,
                seed,
                trial_dir / f"seed{seed}",
                f"s{seed}_" if n_seeds > 1 else "",
            )
            per_seed.append(metrics)
            records.append(record)
        mean = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
        mlflow.log_metrics({f"val_{k}": v for k, v in mean.items()})
        mlflow.log_artifact(str(trial_dir / "config.json"))
        if n_seeds > 1:
            _write_trial_plots(cfgs, trial_dir, records)
    for k, v in mean.items():
        if k not in OBJECTIVES:
            trial.set_user_attr(k, v)
    trial.set_user_attr("trial_dir", str(trial_dir))
    trial.set_user_attr("seeds", seeds)
    values = tuple(mean[k] for k in OBJECTIVES)
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError(f"non-finite objectives {values}")
    print(
        f"trial {trial.number}: "
        + "  ".join(f"{k} {mean[k]:.4g}" for k in OBJECTIVES)
        + f"  (raw MAE {mean['raw_mae']:.1f}, {int(mean['n_params'])} params)",
        flush=True,
    )
    return values


def _storage(url: str) -> str | None:
    """Optuna storage URL; "memory" = in-process only. Creates a sqlite file's folder."""
    if url == "memory":
        return None
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return url


def make_study(cfgs: HPOConfigs) -> optuna.Study:
    hpo = cfgs.hpo
    if hpo.sampler == "tpe":
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(
            multivariate=True, group=True, constant_liar=True, seed=hpo.sampler_seed
        )
    elif hpo.sampler == "nsga2":
        sampler = optuna.samplers.NSGAIISampler(seed=hpo.sampler_seed)
    else:
        raise ValueError(f"unknown sampler {hpo.sampler!r}, expected tpe | nsga2")
    study = optuna.create_study(
        storage=_storage(hpo.storage),
        sampler=sampler,
        study_name=hpo.study_name,
        directions=list(DIRECTIONS),
        load_if_exists=True,
    )
    study.set_metric_names(list(OBJECTIVES))
    return study


def evaluate_trial(
    trial: optuna.trial.FrozenTrial, paired: PairedData, cfgs: HPOConfigs
) -> dict[str, float]:
    """Objectives of a finished trial's saved models on `paired` (mean over seeds)."""
    per_seed = []
    for seed in trial.user_attrs["seeds"]:
        seed_dir = Path(trial.user_attrs["trial_dir"]) / f"seed{seed}"
        model = load_model(seed_dir / "model.pt", map_location=cfgs.train.device)
        pre = Preprocessor.load(seed_dir / "preprocessing.json")
        ds = TaxelDataset(
            paired, np.arange(len(paired.group)), pre.transform_x, pre.transform_y
        ).to(cfgs.train.device)
        per_seed.append(
            objective_metrics(
                predict(model, ds, pre),
                paired.y,
                paired.sensor,
                cfgs.eval.ssim_data_range,
                cfgs.hpo.ssim_window,
                cfgs.hpo.calibrate,
            )
        )
    return {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}


def _plot_pareto(trials: pl.DataFrame, path: Path) -> None:
    """MAE against each other objective; Pareto-optimal trials highlighted."""
    fig = Figure(figsize=(13, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    front = trials["pareto"].to_numpy()
    for ax, name in zip(axes, ("ssim", "pearson", "cosine")):
        x, y = trials["val_mae"].to_numpy(), trials[f"val_{name}"].to_numpy()
        ax.scatter(x[~front], y[~front], s=14, c="0.7", label="trial")
        ax.scatter(x[front], y[front], s=22, c="C3", label="Pareto front")
        ax.set_xlabel("val MAE (TC, calibrated)")
        ax.set_ylabel(f"val {name}")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(path, dpi=130)


def report(study: optuna.Study, data: HPOData, cfgs: HPOConfigs) -> pl.DataFrame:
    """trials.csv, pareto.csv (with test objectives of the saved models), pareto.png."""
    out = _study_dir(cfgs)
    out.mkdir(parents=True, exist_ok=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print("no complete trials yet")
        return pl.DataFrame()
    front = {t.number for t in study.best_trials}
    rows = [
        {
            "trial": t.number,
            "pareto": t.number in front,
            **{f"val_{k}": v for k, v in zip(OBJECTIVES, t.values or [])},
            **{
                f"val_{k}": v
                for k, v in t.user_attrs.items()
                if k.startswith(("raw_", "gain_"))
            },
            "n_params": t.user_attrs.get("n_params"),
            "mlflow_run_id": t.user_attrs.get("mlflow_run_id"),
            **t.params,
        }
        for t in complete
    ]
    # conditional params (taxel_channels) are absent from some rows
    trials = pl.DataFrame(rows, infer_schema_length=None).sort("val_mae")
    trials.write_csv(out / "trials.csv")
    _plot_pareto(trials, out / "pareto.png")

    by_trial = {r["trial"]: r for r in rows}
    best = {t.number: t for t in study.best_trials}
    pareto_rows = []
    for number in trials.filter("pareto")["trial"]:  # sorted by val MAE
        t = best[number]
        test = evaluate_trial(t, data.test, cfgs)
        pareto_rows.append(
            by_trial[t.number] | {f"test_{k}": v for k, v in test.items()}
        )
    pareto = pl.DataFrame(pareto_rows, infer_schema_length=None)
    pareto.write_csv(out / "pareto.csv")
    artifacts.write_json(
        out / "study.json",
        {
            "study_name": study.study_name,
            "objectives": dict(zip(OBJECTIVES, DIRECTIONS)),
            "n_complete": len(complete),
            "n_pareto": len(front),
            "blacklisted_rows": data.n_blacklisted,
            "readings": {
                "train": len(data.train.group),
                "val": len(data.val.group),
                "test": len(data.test.group),
            },
            "hpo": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in asdict(cfgs.hpo).items()
            },
            "note": "test = val files until the real test set is available",
        },
    )
    return pareto


def run_study(cfgs: HPOConfigs) -> pl.DataFrame:
    """Load data, run (or resume) the study, then write its report. Returns the front."""
    data = load_hpo_data(cfgs)
    for name in ("train", "val", "test"):
        pair = getattr(data, name)
        print(
            f"{name}: {len(pair.group)} readings "
            f"({data.n_blacklisted[name]} blacklisted) from {pair.files[0].name}"
        )
    study = make_study(cfgs)
    if not cfgs.hpo.report_only:
        setup_mlflow(cfgs.tracking)
        study.optimize(
            lambda trial: run_trial(trial, data, cfgs),
            n_trials=cfgs.hpo.n_trials,
            timeout=cfgs.hpo.timeout,
            catch=(FloatingPointError, torch.OutOfMemoryError),
        )
    return report(study, data, cfgs)
