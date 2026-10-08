"""Deterministic V0 metrics in raw tactile counts (TC): micro and macro-group.

Micro metrics give every real reading equal weight; macro-group metrics give every raw
(force, Path) condition equal weight (mean over the group's readings first).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
from torch import nn

from taxelnet.dataset import PairedData, TaxelDataset
from taxelnet.preprocessing import Preprocessor


@torch.no_grad()
def predict(
    model: nn.Module, ds: TaxelDataset, pre: Preprocessor, batch_size: int = 1024
) -> np.ndarray:
    """Raw-TC predictions (N, H_taxel, W_taxel) for every row of `ds`, in order."""
    model.eval()
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(ds), batch_size):
        x = ds.x[start : start + batch_size].to(device)
        out.append(pre.inverse_y(model(x)).float().cpu())
    pred = torch.cat(out).numpy()
    if pred.shape != (len(ds), *ds.taxel_shape):
        raise ValueError(
            f"prediction shape {pred.shape} != {(len(ds), *ds.taxel_shape)}"
        )
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
    return pred


def _group_mean(per_reading: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Mean of `per_reading` (N, ...) within each group present, (G, ...)."""
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), *per_reading.shape[1:]))
    np.add.at(sums, inverse, per_reading)
    return sums / counts.reshape(-1, *[1] * (per_reading.ndim - 1))


def _micro_macro(name: str, per_reading: np.ndarray, group: np.ndarray) -> dict:
    return {
        f"micro_{name}": float(per_reading.mean()),
        f"macro_{name}": float(_group_mean(per_reading, group).mean()),
    }


def global_ssim(pred: np.ndarray, target: np.ndarray, data_range: float) -> np.ndarray:
    """SSIM with one window covering each whole map; (N,). Not a training loss."""
    p = pred.reshape(len(pred), -1).astype(np.float64)
    t = target.reshape(len(target), -1).astype(np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mp, mt = p.mean(1), t.mean(1)
    vp, vt = p.var(1), t.var(1)
    cov = ((p - mp[:, None]) * (t - mt[:, None])).mean(1)
    return ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp**2 + mt**2 + c1) * (vp + vt + c2)
    )


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, group: np.ndarray, ssim_data_range: float
) -> dict:
    """Point metrics in raw TC. For a deterministic model CRPS = MAE and the energy score
    is the Euclidean error, so those are reported for the shared metric ladder."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {pred.shape} and target {target.shape} differ")
    n = len(pred)
    r = (pred - target).astype(np.float64).reshape(n, -1)
    abs_r, sq_r = np.abs(r), r**2

    m: dict[str, Any] = {"n_readings": n, "n_groups": len(np.unique(group))}
    m |= _micro_macro("mae", abs_r.mean(1), group)
    m |= _micro_macro("mse", sq_r.mean(1), group)
    m["micro_rmse"] = float(np.sqrt(m["micro_mse"]))
    m["macro_rmse"] = float(np.sqrt(m["macro_mse"]))
    m["micro_crps"], m["macro_crps"] = m["micro_mae"], m["macro_mae"]
    m |= _micro_macro("energy_score", np.sqrt(sq_r.sum(1)), group)
    m |= _micro_macro("ssim", global_ssim(pred, target, ssim_data_range), group)
    m["ssim_data_range"] = ssim_data_range
    m["per_taxel_micro_mae"] = abs_r.mean(0).tolist()
    m["per_taxel_micro_rmse"] = np.sqrt(sq_r.mean(0)).tolist()
    m["per_taxel_macro_mae"] = _group_mean(abs_r, group).mean(0).tolist()
    m["per_taxel_macro_rmse"] = np.sqrt(_group_mean(sq_r, group).mean(0)).tolist()
    return m


def reference_points(target: np.ndarray, group: np.ndarray) -> dict:
    """Replicate noise floor and trivial zero predictor, on the evaluated partition.

    Noise floor: MAE of the readings of repeated groups (n >= 2) against their group's
    per-taxel median.
    """
    n = len(target)
    t = target.astype(np.float64).reshape(n, -1)
    _, inverse, counts = np.unique(group, return_inverse=True, return_counts=True)
    repeated = counts[inverse] >= 2
    out = {
        "trivial_zero": _micro_macro("mae", np.abs(t).mean(1), group),
        "noise_floor": {"n_repeated_groups": int((counts >= 2).sum())},
    }
    if repeated.any():
        tr, gr = t[repeated], inverse[repeated]
        medians = np.zeros((len(counts), t.shape[1]))
        for g in np.unique(gr):
            medians[g] = np.median(tr[gr == g], axis=0)
        out["noise_floor"] |= _micro_macro("mae", np.abs(tr - medians[gr]).mean(1), gr)
        out["noise_floor"]["n_readings"] = int(repeated.sum())
    return out


def predictions_frame(
    paired: PairedData, rows: np.ndarray, partition: str, pred: np.ndarray
) -> pl.DataFrame:
    """One record per real target: raw key, partition, provenance, target, pred, residual."""
    target = paired.y[rows].reshape(len(rows), -1)
    flat_pred = pred.reshape(len(rows), -1)
    groups = [paired.groups[g] for g in paired.group[rows]]
    return pl.DataFrame(
        {
            "force": [g["force"] for g in groups],
            "path": [g["path"] for g in groups],
            "dataset": [g["dataset"] for g in groups],
            "partition": [partition] * len(rows),
            "source_row": paired.source_rows[rows],
            "target": target,
            "prediction": flat_pred,
            "residual": flat_pred - target,
        }
    )
