"""Saturating output head and compressive input transform (experiment `sat`, plan C2).

ReLU networks extrapolate linearly outside the training support, so deep test contacts
are read as proportionally more signal (V0 predicts up to 5,911 TC on test while the
training maximum is 2,711 TC). Two independent remedies, both off by default:

- `BoundedTanh` head (`model.head_activation = "tanh_bounded"`): an asymmetric soft clip
  `y = hi * tanh(z / hi)` for z >= 0 and `y = lo * tanh(z / lo)` for z < 0 (lo < 0), in
  the scaled target space. Identity slope at 0, C1, bounded to (lo, hi). The bounds are
  derived from the train target range times a margin and stored as module buffers, so
  they travel with every state_dict (best.pt, last.pt, model.pt).
- `asinh` input scaler: `x' = asinh(x / s) / asinh(x_ref / s)` with x_ref the train
  max |x|. Linear for |x| << s, logarithmic for |x| >> s; one global scalar transform,
  so the convolutions stay translation-equivariant.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

HEAD_ACTIVATIONS = ("linear", "tanh_bounded")


class BoundedTanh(nn.Module):
    """Asymmetric soft clip to (lo, hi), identity slope at 0."""

    def __init__(self, lo: float = -1.0, hi: float = 1.0) -> None:
        super().__init__()
        if not (lo < 0 < hi):
            raise ValueError(f"bounds must satisfy lo < 0 < hi, got {lo}, {hi}")
        self.register_buffer("lo", torch.tensor(float(lo)))
        self.register_buffer("hi", torch.tensor(float(hi)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(
            z >= 0, self.hi * torch.tanh(z / self.hi), self.lo * torch.tanh(z / self.lo)
        )

    def extra_repr(self) -> str:
        return f"lo={self.lo.item():.4g}, hi={self.hi.item():.4g}"


def head_bounds(y_scaled_train: np.ndarray, margin: float) -> tuple[float, float]:
    """(lo, hi) = margin * (min, max) of the scaled train targets, kept away from 0."""
    y = np.asarray(y_scaled_train, dtype=np.float64)
    lo, hi = margin * float(y.min()), margin * float(y.max())
    eps = 1e-3 * max(abs(lo), abs(hi), 1.0)
    return min(lo, -eps), max(hi, eps)


def asinh_compress(x, scale: float):
    """asinh(x / scale) for numpy arrays or torch tensors."""
    if isinstance(x, torch.Tensor):
        return torch.asinh(x / scale)
    return np.arcsinh(x / scale)
