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
"""

from __future__ import annotations

import copy
import gc
import math
import warnings
from dataclasses import asdict, dataclass
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
from taxelnet.checkpoint import load_model, save_model
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
    cosine_similarity,
    pearson,
    predict,
    windowed_ssim,
)
from taxelnet.model import build_model
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
    labels = np.full(len(pred), "all") if pads is None else np.asarray(pads).astype(str)
    cal = pred.astype(np.float64)
    out: dict[str, float] = {}
    if calibrate:
        cal = cal.copy()
        for q in sorted(set(labels)):
            sel = labels == q
            gain = float(fit_global_gain(pred[sel], target[sel]).gain)
            cal[sel] *= gain
            out[f"gain_{q}"] = gain
    out |= {
        "ssim": float(windowed_ssim(cal, target, data_range, window).mean()),
        "pearson": float(pearson(cal, target).mean()),
        "mae": float(np.abs(cal - target).mean()),
        "cosine": float(cosine_similarity(cal, target).mean()),
        "raw_ssim": float(windowed_ssim(pred, target, data_range, window).mean()),
        "raw_mae": float(np.abs(pred.astype(np.float64) - target).mean()),
    }
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
    t.scheduler = "cosine"
    p.target_group_key = trial.suggest_categorical(
        "target_group_key", ["none", "sensor"]
    )
    p.target_scaler, p.input_scaler = "global_scale", "max_abs"
    return cfgs


def _study_dir(cfgs: HPOConfigs) -> Path:
    return cfgs.hpo.output_dir / cfgs.hpo.study_name


def _train_seed(
    cfgs: HPOConfigs, data: HPOData, seed: int, out_dir: Path, log_prefix: str
) -> dict[str, Any]:
    """Train one model on the whole training pair; save its bundle; val objectives."""
    _seed_everything(seed)
    train_cfg = copy.deepcopy(cfgs.train)
    train_cfg.seed, train_cfg.test_every = seed, 0
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
    finally:
        del model, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {**metrics, "n_params": n_params}


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
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            cfgs.flat() | {f"trial.{k}": v for k, v in trial.params.items()}
        )
        mlflow.set_tags(
            {
                "optuna.study": cfgs.hpo.study_name,
                "optuna.trial": str(trial.number),
                "git.commit": artifacts.git_revision(Path(__file__).parent)["commit"],
            }
        )
        per_seed = [
            _train_seed(
                cfgs,
                data,
                seed,
                trial_dir / f"seed{seed}",
                f"s{seed}_" if n_seeds > 1 else "",
            )
            for seed in seeds
        ]
        mean = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0]}
        mlflow.log_metrics({f"val_{k}": v for k, v in mean.items()})
        mlflow.log_artifact(str(trial_dir / "config.json"))
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
