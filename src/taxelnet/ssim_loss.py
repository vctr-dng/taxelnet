"""Differentiable windowed SSIM on taxel maps, for the hybrid L1 / SSIM training loss.

L = alpha * L1 + (1 - alpha) * (1 - SSIM)

SSIM is the torch twin of `taxelnet.evaluation.windowed_ssim`: mean over every `window`
patch (uniform weights, stride 1) of each map, population statistics, and the constants
c1 = (0.01 R)^2, c2 = (0.03 R)^2 of the fixed data range R. The training loop feeds it
raw-TC maps with R = `train.loss_ssim_data_range` (3600 TC, the evaluation range), so the
term is the HPO SSIM objective before pad calibration.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def windowed_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
    window: tuple[int, int] = (2, 2),
) -> torch.Tensor:
    """Per-reading mean SSIM over the `window` patches of (N, H, W) maps; (N,)."""
    wh, ww = window
    h, w = pred.shape[-2:]
    if not (1 <= wh <= h and 1 <= ww <= w):
        raise ValueError(f"window {window} does not fit the {h} x {w} map")
    # (N, wh * ww, n_patches)
    p = F.unfold(pred.unsqueeze(1), window)
    t = F.unfold(target.unsqueeze(1), window)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mp, mt = p.mean(1, keepdim=True), t.mean(1, keepdim=True)
    dp, dt = p - mp, t - mt
    vp, vt, cov = (dp**2).mean(1), (dt**2).mean(1), (dp * dt).mean(1)
    mp, mt = mp.squeeze(1), mt.squeeze(1)
    ssim = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp**2 + mt**2 + c1) * (vp + vt + c2)
    )
    return ssim.mean(1)
