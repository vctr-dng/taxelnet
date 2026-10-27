"""Deterministic V0 metrics in raw tactile counts (TC): micro and macro-group.

Micro metrics give every real reading equal weight; macro-group metrics give every raw
(force, Path) condition equal weight (mean over the group's readings first).

Decomposed metrics split the error into pattern, amplitude and offset parts (per map):
cosine(pred, target); log(||pred|| / ||target||); MAE after least-squares scale + shift
alignment (MiDaS-style) and after one global least-squares gain; MAE after removing the
map's median residual; argmax-taxel hit and grid distance; IoU of the contact sets.
Distribution summaries (`*_median`, `*_iqr`) are over readings; `macro_*_median` is over
group means.
"""

from __future__ import annotations

import re
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
        sensor = None if ds.sensor is None else ds.sensor[start : start + batch_size]
        out.append(pre.inverse_y(model(x), group=sensor).float().cpu())
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


def windowed_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float,
    window: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Mean SSIM over every `window` patch (uniform weights, stride 1) of each map; (N,).

    Same constants and population statistics as `global_ssim`, which is the special case
    of a window covering the whole map. A 2 x 2 window on a 7 x 4 map has 6 x 3 patches.
    """
    wh, ww = window
    h, w = pred.shape[-2:]
    if not (1 <= wh <= h and 1 <= ww <= w):
        raise ValueError(f"window {window} does not fit the {h} x {w} map")
    view = np.lib.stride_tricks.sliding_window_view
    # (N, h - wh + 1, w - ww + 1, wh * ww)
    p = view(pred.astype(np.float64), window, axis=(-2, -1)).reshape(
        len(pred), h - wh + 1, w - ww + 1, -1
    )
    t = view(target.astype(np.float64), window, axis=(-2, -1)).reshape(p.shape)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mp, mt = p.mean(-1), t.mean(-1)
    vp, vt = p.var(-1), t.var(-1)
    cov = ((p - mp[..., None]) * (t - mt[..., None])).mean(-1)
    ssim = ((2 * mp * mt + c1) * (2 * cov + c2)) / (
        (mp**2 + mt**2 + c1) * (vp + vt + c2)
    )
    return ssim.mean((1, 2))


def pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation over the taxels of each map; 0 when either map is flat. (N,)"""
    p = pred.reshape(len(pred), -1).astype(np.float64)
    t = target.reshape(len(target), -1).astype(np.float64)
    pc, tc = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    denom = np.sqrt((pc**2).sum(1) * (tc**2).sum(1))
    return np.where(denom > 1e-12, (pc * tc).sum(1) / np.maximum(denom, 1e-12), 0.0)


def cosine_similarity(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Cosine of the angle between each predicted and real map (taxel vectors); (N,)."""
    p = pred.reshape(len(pred), -1).astype(np.float64)
    t = target.reshape(len(target), -1).astype(np.float64)
    pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
    return (p * t).sum(1) / np.maximum(pn * tn, 1e-12)


CONTACT_THRESHOLD = 50.0  # TC: a taxel above this is "in contact" for contact_iou
NORM_FLOOR = 1.0  # TC: map norms are floored at this in the amplitude log-ratio


def _median_iqr(name: str, per_reading: np.ndarray, group: np.ndarray) -> dict:
    q25, q50, q75 = np.percentile(per_reading, [25, 50, 75])
    return {
        f"{name}_median": float(q50),
        f"{name}_iqr": float(q75 - q25),
        f"macro_{name}_median": float(np.median(_group_mean(per_reading, group))),
    }


def decomposed_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    contact_threshold: float = CONTACT_THRESHOLD,
) -> dict:
    """Pattern / amplitude / offset decomposition of the error, raw TC, per reading."""
    n = len(pred)
    p = pred.astype(np.float64).reshape(n, -1)
    t = target.astype(np.float64).reshape(n, -1)
    m: dict[str, Any] = {}

    pn, tn = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
    m |= _median_iqr("cosine", cosine_similarity(p, t), group)
    logratio = np.log(np.maximum(pn, NORM_FLOOR) / np.maximum(tn, NORM_FLOOR))
    m |= _median_iqr("amp_logratio", logratio, group)

    # per-map least-squares scale + shift of pred onto target
    pc, tc = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    var = (pc**2).sum(1)
    scale = np.where(var > 1e-12, (pc * tc).sum(1) / np.maximum(var, 1e-12), 0.0)
    aligned = scale[:, None] * pc + t.mean(1, keepdims=True)
    m |= _micro_macro("aligned_mae", np.abs(aligned - t).mean(1), group)

    # one least-squares gain (no shift) for the whole partition
    pp = (p**2).sum()
    gain = float((p * t).sum() / pp) if pp > 0 else 0.0
    m["gain_aligned_gain"] = gain
    m |= _micro_macro("gain_aligned_mae", np.abs(gain * p - t).mean(1), group)

    r = p - t
    r_off = r - np.median(r, axis=1, keepdims=True)
    m |= _micro_macro("offset_removed_mae", np.abs(r_off).mean(1), group)

    width = target.shape[-1]
    ap, at = p.argmax(1), t.argmax(1)
    m |= _micro_macro("peak_hit", (ap == at).astype(np.float64), group)
    dist = np.hypot(ap // width - at // width, ap % width - at % width)
    m |= _micro_macro("peak_dist", dist.astype(np.float64), group)

    cp, ct = p > contact_threshold, t > contact_threshold
    union = (cp | ct).sum(1)
    iou = np.where(union > 0, (cp & ct).sum(1) / np.maximum(union, 1), 1.0)
    m |= _micro_macro("contact_iou", iou, group)
    return m


def object_group(name: str) -> str:
    """Test object group of a grasp `Name`: tokens 0..2, e.g. Cylinder_TestButtons_235."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", "_".join(name.split("_")[:3]))


def object_metrics(
    pred: np.ndarray, target: np.ndarray, group: np.ndarray, names: list[str]
) -> dict:
    """Macro MAE per test object group and without the cylinders; zero-predictor refs.

    `names` is the grasp `Name` of every reading. Returns flat float entries
    `group_<object>_mae`, `nocyl_micro_mae`, `nocyl_macro_mae` and a `reference` dict
    with the same entries for the all-zero predictor.
    """
    n = len(pred)
    err = np.abs((pred - target).astype(np.float64).reshape(n, -1)).mean(1)
    zero = np.abs(target.astype(np.float64).reshape(n, -1)).mean(1)
    objects = np.array([object_group(s) for s in names])
    out: dict[str, Any] = {}
    ref: dict[str, Any] = {}
    for obj in sorted(set(objects)):
        sel = objects == obj
        out[f"group_{obj}_mae"] = float(_group_mean(err[sel], group[sel]).mean())
        ref[f"group_{obj}_mae"] = float(_group_mean(zero[sel], group[sel]).mean())
    nocyl = ~np.char.startswith(objects.astype(str), "Cylinder")
    if nocyl.any():
        for d, e in ((out, err), (ref, zero)):
            mm = _micro_macro("mae", e[nocyl], group[nocyl])
            d |= {f"nocyl_{k}": v for k, v in mm.items()}
    out["reference"] = ref
    return out


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
    m |= decomposed_metrics(pred, target, group)
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
    """One record per real target: raw key, partition, provenance, target, pred, residual.

    `path` is the second key column (`Path` in training files, `Name` in the test file).
    """
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
