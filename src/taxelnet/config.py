"""Typed, serializable configuration sections + `--<section>.<field>` CLI overrides."""

from __future__ import annotations

import argparse
import dataclasses
import os
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class DataConfig:
    # training pair: split by (force, Path) group into train and val
    data_dir: Path = Path("data/newton/v1/training")
    real_csv: str = "Dataset_Indenter_Real.csv"
    sim_csv: str = "Dataset_Indenter_Simulation.csv"
    key_cols: tuple[str, str] = ("force", "Path")
    val_ratio: float = 0.1
    split_seed: int = 0
    # group: (force, Path) groups split independently (force interpolation on seen
    # shapes); shape: whole indenter shapes (Path before "/") held out in val
    split_strategy: str = "group"
    # shape_fold: k-fold over whole shapes, `fold` (0-based) is the val fold
    n_folds: int = 5
    fold: int = 0
    # external test pair (grasp export): evaluated only, never split or fitted on
    test_dir: Path = Path("data/newton/v1/test")
    test_real_csv: str = "Dataset_Grasp_Real_qtac100_testing.csv"
    test_sim_csv: str = "Dataset_Grasp_Simulation_qtac100_testing.csv"
    test_key_cols: tuple[str, str] = ("force", "Name")
    # parsed simulation grids, keyed by CSV checksum (shared by train and test files)
    cache_dir: Path | None = Path("data/newton/v1/training/.cache")


@dataclass
class PreprocessingConfig:
    # one global scalar for the whole grid: keeps the convolutions translation-equivariant
    input_scaler: str = "max_abs"  # identity | max_abs | zscore | robust_zscore | asinh
    # see the head comment in taxelnet.model for the trade-offs between these three
    # identity | global_scale | per_taxel_zscore | asinh
    target_scaler: str = "global_scale"
    # optional [min, max] in mm; never clip at 0 by default (negatives are pad bulge)
    clip_input: tuple[float, float] | None = None
    # asinh target only: y' = asinh(y_tc / asinh_scale); the inverse clamps y' to the
    # train range of y' widened by asinh_clamp_margin (asinh units) on each side
    asinh_scale: float = 100.0
    asinh_clamp_margin: float = 0.25
    # asinh only: x' = asinh(x / s) / asinh(max|x_train| / s), s in mm (taxelnet.saturation)
    input_asinh_scale: float = 0.2
    # none | sensor: fit the target scaler (global_scale | per_taxel_zscore only)
    # separately per pad label (S1/S2, the Path/Name suffix) and invert with each row's own
    # parameters. The network input never sees the label.
    target_group_key: str = "none"


@dataclass
class ModelConfig:
    name: str = "cnn_v0"
    base_channels: int = 16
    num_down_blocks: int = 3
    # learned per-taxel gain/offset on the output (y_t = g_t * z_t + o_t), init identity
    taxel_affine_head: bool = False
    # linear | tanh_bounded: soft clip of the scaled output to (lo, hi) (taxelnet.saturation)
    head_activation: str = "linear"
    # tanh_bounded: bounds = margin * (min, max) of the scaled train targets, filled in
    # at fit time unless given explicitly (scaled target units, not TC)
    head_bound_margin: float = 1.1
    head_lower: float | None = None
    head_upper: float | None = None
    # cnn_feat, and the footprint_cnn residual when residual_backbone="cnn_feat"
    # (taxelnet.model_features): comma list of z, lap, grad, coord
    input_features: str = "z"
    convs_per_block: int = 1
    norm: str = "none"  # none | group (GroupNorm)
    # footprint_cnn only: CNN residual branch, cnn_v0 | cnn_feat (uses the three
    # cnn_feat knobs above)
    residual_backbone: str = "cnn_v0"


@dataclass
class TrainConfig:
    epochs: int = 300
    groups_per_batch: int = 64  # batches hold whole (force, Path) groups
    lr: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine | none
    # group weight n_g ** alpha: 0 = group-balanced (default), 1 = micro (ablation only)
    group_weight_alpha: float = 0.0
    # auxiliary |log(||y_hat||+eps) - log(||y||+eps)| per reading (norm over taxels, raw
    # TC), group-balanced like L1 and added with this weight; 0 = off (taxelnet.amp)
    loss_logamp_weight: float = 0.0
    loss_logamp_eps: float = 50.0  # TC
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed: int = 0
    output_dir: Path = Path("outputs")
    # every N epochs (and at the last epoch): save checkpoints/epoch_<N>.pt, log test_*
    # metrics of that model, and inference plots. 0 disables. Monitoring only: best.pt is
    # always selected on validation, never on test.
    test_every: int = 10
    # L1 penalty on the mean |CNN residual| (scaled target units) of hybrid models that
    # expose `model.residual` (footprint_cnn); 0 disables
    residual_l1_weight: float = 0.0
    # separate lr for the physics parameters (`model.footprint`) of hybrid models;
    # None = train.lr for every parameter
    physics_lr: float | None = None
    # ssi experiment: L = L1 + loss_ssi_weight * L_ssi + loss_grad_weight * L_grad, the
    # MiDaS scale-and-shift-invariant pattern terms (taxelnet.ssi), group-balanced like
    # L1. 0 = off (default). loss_ssi_eps: MAD floor of the per-map normalization, in
    # scaled target units.
    loss_ssi_weight: float = 0.0
    loss_grad_weight: float = 0.0
    loss_ssi_eps: float = 0.05
    # warm start (fine-tuning): run bundle whose model.pt weights initialise the model
    # (same architecture) and whose input scaler (x_center, x_scale) is reused, so the
    # input means the same as in pre-training; the target scaler is refitted on this
    # run's training rows. None = random init.
    init_from: Path | None = None


@dataclass
class EvalConfig:
    # fixed SSIM data range in TC, spanning the real target range (about -888 to 2,711)
    ssim_data_range: float = 3600.0
    plot_samples: int = 6  # test readings drawn in the inference plots
    # readings drawn per (object, force, pad) group in plots/<name>/samples/<group>.png
    plot_samples_per_group: int = 3


@dataclass
class TrackingConfig:
    # None -> $MLFLOW_TRACKING_URI, else a local sqlite store; remote URIs are explicit
    uri: str | None = None
    experiment: str = "taxelnet-v0"
    run_name: str | None = None
    # used only when the experiment is created; None = the tracking server's default
    artifact_location: str | None = None

    def resolved_uri(self) -> str:
        return (
            self.uri or os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
        )


@dataclass
class Configs:
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {
            f.name: {
                k: _jsonable(v)
                for k, v in dataclasses.asdict(getattr(self, f.name)).items()
            }
            for f in dataclasses.fields(self)
        }

    def flat(self) -> dict[str, Any]:
        return {f"{s}.{k}": v for s, d in self.to_dict().items() for k, v in d.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _parser_for(annotation: Any):
    """argparse `type=` for a dataclass field annotation (None is spelled 'none')."""
    args = typing.get_args(annotation)
    optional = typing.get_origin(annotation) in (typing.Union, types.UnionType) and (
        type(None) in args
    )
    base = next(a for a in args if a is not type(None)) if optional else annotation

    def parse(text: str):
        if optional and text.lower() == "none":
            return None
        if typing.get_origin(base) is tuple:
            item_types = typing.get_args(base)
            items = text.split(",")
            if len(items) != len(item_types):
                raise argparse.ArgumentTypeError(
                    f"expected {len(item_types)} comma values"
                )
            return tuple(t(v) for t, v in zip(item_types, items))
        if base is bool:
            return text.lower() in ("1", "true", "yes")
        return base(text)

    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Taxelnet V0 model.")
    defaults = Configs()
    for section in dataclasses.fields(Configs):
        cfg = getattr(defaults, section.name)
        hints = typing.get_type_hints(type(cfg))
        for f in dataclasses.fields(cfg):
            parser.add_argument(
                f"--{section.name}.{f.name}",
                dest=f"{section.name}.{f.name}",
                type=_parser_for(hints[f.name]),
                default=argparse.SUPPRESS,
                help=f"default: {_jsonable(getattr(cfg, f.name))}",
            )
    return parser


def parse_configs(argv: list[str] | None = None) -> Configs:
    args = vars(build_arg_parser().parse_args(argv))
    cfgs = Configs()
    for dotted, value in args.items():
        section, name = dotted.split(".", 1)
        setattr(getattr(cfgs, section), name, value)
    return cfgs
