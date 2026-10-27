"""Scale-and-shift-invariant pattern losses (MiDaS, Ranftl et al. 2019), per reading.

Each 7x4 map d is normalized separately for prediction and target:

    d_hat = (d - median(d)) / (mean|d - median(d)| + eps)

L_ssi  = mean over taxels |d_hat_pred - d_hat_target|               (MiDaS L_ssimae)
L_grad = mean |dR/dx| + mean |dR/dy|,  R = d_hat_pred - d_hat_target  (single-scale
         gradient matching with finite differences on the taxel grid)

Both are returned per reading, so the caller can group-balance them like the L1 anchor.
`eps` is in scaled target units (default 0.05 = ~23 TC with `global_scale`, the
replicate noise floor). It keeps near-flat maps from amplifying noise: a target map whose
MAD is << eps normalizes to ~0 instead of to unit spread, and the 1/(MAD + eps) factor
bounds the gradient through the prediction's normalization by 1/eps.
"""

from __future__ import annotations

import torch


def ssi_normalize(d: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-map (dim 0 = readings) median shift and mean-absolute-deviation scale."""
    flat = d.flatten(1)
    med = flat.median(dim=1, keepdim=True).values
    centered = flat - med
    scale = centered.abs().mean(dim=1, keepdim=True) + eps
    return (centered / scale).view_as(d)


def ssi_losses(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-reading (L_ssi, L_grad) for (N, H, W) prediction and target maps."""
    residual = ssi_normalize(pred, eps) - ssi_normalize(target, eps)
    l_ssi = residual.abs().flatten(1).mean(1)
    dx = (residual[..., :, 1:] - residual[..., :, :-1]).abs().flatten(1).mean(1)
    dy = (residual[..., 1:, :] - residual[..., :-1, :]).abs().flatten(1).mean(1)
    return l_ssi, dx + dy
