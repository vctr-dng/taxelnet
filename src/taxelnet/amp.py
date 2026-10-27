"""Amplitude experiment: log-amplitude auxiliary loss and the asinh target transform.

- `logamp_per_press`: |log(||y_hat||_2 + eps) - log(||y||_2 + eps)| per reading, the norm
  taken over all taxels in raw TC (scale-invariant log loss of Eigen et al.). The caller
  group-balances it like L1 (`training.group_mean`).
- asinh target transform: y' = asinh(y / s); inverse s * sinh(clamp(y', lo, hi)), with
  [lo, hi] the train range of y' widened by a margin so sinh cannot explode when the
  model extrapolates. L1 in y' space still estimates the per-taxel median in raw TC
  (quantiles are equivariant under monotone maps) but rebalances gradients between quiet
  and loud taxels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def logamp_per_press(
    pred_tc: torch.Tensor, target_tc: torch.Tensor, eps: float
) -> torch.Tensor:
    """(B, H, W) raw-TC predictions/targets -> (B,) absolute log-amplitude error."""
    norm_p = pred_tc.flatten(1).norm(dim=1)
    norm_t = target_tc.flatten(1).norm(dim=1)
    return (torch.log(norm_p + eps) - torch.log(norm_t + eps)).abs()


def asinh_forward(y: Any, scale: float) -> Any:
    """Raw TC -> asinh space (numpy or torch)."""
    if isinstance(y, torch.Tensor):
        return torch.asinh(y / scale)
    return np.arcsinh(y / scale)


def asinh_inverse(
    y: Any, scale: float, lo: float | None = None, hi: float | None = None
) -> Any:
    """asinh space -> raw TC; clamps to [lo, hi] first when the bounds are given."""
    if lo is not None and hi is not None:
        y = y.clip(lo, hi)
    if isinstance(y, torch.Tensor):
        return scale * torch.sinh(y)
    return scale * np.sinh(y)


def asinh_clamp_bounds(
    y_train_tc: np.ndarray, scale: float, margin: float
) -> tuple[float, float]:
    """Train range of asinh(y / s), widened by `margin` (in asinh units) on each side."""
    t = asinh_forward(np.asarray(y_train_tc, dtype=np.float64), scale)
    return float(t.min() - margin), float(t.max() + margin)
