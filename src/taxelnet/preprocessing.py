"""Train-only fitted input/target transforms and the tensor-contract metadata.

Scalers are fitted on training rows only and applied unchanged to validation, test and
inference. Metrics are always computed in raw tactile counts (TC) after `inverse_y`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from taxelnet import amp
from taxelnet.config import PreprocessingConfig
from taxelnet.io_csv import (
    SIM_GRID_SHAPE,
    TAXEL_GRID_SHAPE,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

INPUT_SCALERS = ("identity", "max_abs", "zscore", "robust_zscore")
TARGET_SCALERS = ("identity", "global_scale", "per_taxel_zscore", "asinh")

Array = TypeVar("Array", np.ndarray, torch.Tensor)


def tensor_contract() -> dict[str, Any]:
    """Units, sign, orientation and deployment policy of the model input and output."""
    n_rows, n_cols = SIM_GRID_SHAPE
    return {
        "input": {
            "quantity": "Z deformation of the 3,773 top-layer pad nodes",
            "units": "mm",
            "sign": "positive = pushed down; Z(rest frame) - Z(last HOLD frame)",
            "grid_shape": list(SIM_GRID_SHAPE),
            "layout": "(B, 1, H, W) row-major; node order of io_csv.sim_node_columns()",
            "node_ids": "node_ids.csv",
            "pad_extent_mm": {"x": [X_MIN, X_MAX], "y": [Y_MAX, Y_MIN]},
            "node_spacing_mm": {
                "x": (X_MAX - X_MIN) / (n_rows - 1),
                "y": (Y_MAX - Y_MIN) / (n_cols - 1),
            },
            "simulator": "Newton 1.5",
            "pad_model": "QTAC100",
            "rest_frame_policy": "first frame (pad at rest) replaces a no-contact baseline",
            "hold_frame_policy": "deformation read at the last HOLD frame, at target force",
            "rigid_alignment": "undefined: deployment compatibility not claimed",
        },
        "output": {
            "quantity": "baseline-corrected physical taxel reading",
            "units": "tactile counts (TC)",
            "grid_shape": list(TAXEL_GRID_SHAPE),
            "layout": "(B, H, W) row-major; tax1..tax4 is the first row",
        },
        "orientation": (
            "tax1 and n195 are top-left at X = 41.80 mm, Y = -1.50 mm; rows run along X "
            "(pad length), columns along Y (width); no flip between the two grids"
        ),
    }


@dataclass
class Preprocessor:
    config: PreprocessingConfig
    # input: x' = (clip(x) - x_center) / x_scale, global scalars
    x_center: float = 0.0
    x_scale: float = 1.0
    # target: y' = (y - y_center) / y_scale, scalars or (H, W) per-taxel arrays
    y_center: np.ndarray = field(default_factory=lambda: np.zeros(()))
    y_scale: np.ndarray = field(default_factory=lambda: np.ones(()))
    # asinh only: y' = asinh(y / asinh_scale); inverse clamps y' to these train bounds
    y_clamp_lo: float | None = None
    y_clamp_hi: float | None = None
    fitted_on: str | None = None
    n_fit_rows: int = 0

    @classmethod
    def fit(
        cls, config: PreprocessingConfig, x_train: np.ndarray, y_train: np.ndarray
    ) -> Preprocessor:
        """Fit on training rows only: x_train (N, H, W) mm, y_train (N, h, w) TC."""
        if config.input_scaler not in INPUT_SCALERS:
            raise ValueError(f"input_scaler must be one of {INPUT_SCALERS}")
        if config.target_scaler not in TARGET_SCALERS:
            raise ValueError(f"target_scaler must be one of {TARGET_SCALERS}")
        pre = cls(config, fitted_on="train", n_fit_rows=len(x_train))

        x = pre._clip(np.asarray(x_train, dtype=np.float64))
        match config.input_scaler:
            case "max_abs":
                pre.x_scale = float(np.abs(x).max())
            case "zscore":
                pre.x_center, pre.x_scale = float(x.mean()), float(x.std())
            case "robust_zscore":
                q25, med, q75 = np.percentile(x, [25, 50, 75])
                pre.x_center, pre.x_scale = float(med), float(q75 - q25)
        if not pre.x_scale > 0:
            raise ValueError(f"degenerate input scale {pre.x_scale}")

        y = np.asarray(y_train, dtype=np.float64)
        match config.target_scaler:
            case "global_scale":
                pre.y_scale = np.array(y.std())
            case "per_taxel_zscore":
                pre.y_center, pre.y_scale = y.mean(axis=0), y.std(axis=0)
            case "asinh":
                if not config.asinh_scale > 0:
                    raise ValueError(f"asinh_scale must be > 0, got {config.asinh_scale}")
                pre.y_clamp_lo, pre.y_clamp_hi = amp.asinh_clamp_bounds(
                    y, config.asinh_scale, config.asinh_clamp_margin
                )
        if not (pre.y_scale > 0).all():
            raise ValueError("degenerate target scale")
        return pre

    def _clip(self, x: Array) -> Any:
        if self.config.clip_input is None:
            return x
        lo, hi = self.config.clip_input
        return x.clip(lo, hi)

    def _like(self, value: np.ndarray, ref: Array) -> Any:
        if isinstance(ref, torch.Tensor):
            return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
        return value.astype(ref.dtype)

    def transform_x(self, x: Array) -> Array:
        return (self._clip(x) - self.x_center) / self.x_scale

    def transform_y(self, y: Array) -> Array:
        if self.config.target_scaler == "asinh":
            return amp.asinh_forward(y, self.config.asinh_scale)
        return (y - self._like(self.y_center, y)) / self._like(self.y_scale, y)

    def inverse_y(self, y: Array, clamp: bool = True) -> Array:
        if self.config.target_scaler == "asinh":
            lo, hi = (self.y_clamp_lo, self.y_clamp_hi) if clamp else (None, None)
            return amp.asinh_inverse(y, self.config.asinh_scale, lo, hi)
        return y * self._like(self.y_scale, y) + self._like(self.y_center, y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                **asdict(self.config),
                "clip_input": (
                    list(self.config.clip_input) if self.config.clip_input else None
                ),
            },
            "fitted": {
                "fitted_on": self.fitted_on,
                "n_fit_rows": self.n_fit_rows,
                "x_center": self.x_center,
                "x_scale": self.x_scale,
                "y_center": np.asarray(self.y_center).tolist(),
                "y_scale": np.asarray(self.y_scale).tolist(),
                "y_clamp_lo": self.y_clamp_lo,
                "y_clamp_hi": self.y_clamp_hi,
            },
            "formulas": {
                "input": "x_scaled = (clip(x_mm) - x_center) / x_scale",
                "target": "y_scaled = (y_tc - y_center) / y_scale",
                "inverse": "y_tc = y_scaled * y_scale + y_center",
                "asinh_target": "y_scaled = asinh(y_tc / asinh_scale)",
                "asinh_inverse": (
                    "y_tc = asinh_scale * sinh(clip(y_scaled, y_clamp_lo, y_clamp_hi))"
                ),
            },
            "contract": tensor_contract(),
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        data = json.loads(path.read_text())
        cfg = dict(data["config"])
        if cfg["clip_input"] is not None:
            cfg["clip_input"] = tuple(cfg["clip_input"])
        fitted = data["fitted"]
        return cls(
            PreprocessingConfig(**cfg),
            x_center=fitted["x_center"],
            x_scale=fitted["x_scale"],
            y_center=np.asarray(fitted["y_center"]),
            y_scale=np.asarray(fitted["y_scale"]),
            y_clamp_lo=fitted.get("y_clamp_lo"),
            y_clamp_hi=fitted.get("y_clamp_hi"),
            fitted_on=fitted["fitted_on"],
            n_fit_rows=fitted["n_fit_rows"],
        )
