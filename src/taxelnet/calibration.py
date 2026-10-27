"""Few-shot affine output calibration: a post-hoc evaluation protocol in raw TC.

A trained model's prediction p (N, T) is corrected on a small calibration subset of a
target domain (an object, a session) and evaluated on the rest of that domain:

- ``global_gain``:   y ~ g * p
- ``global_affine``: y ~ g * p + o              (o: one scalar offset per map)
- ``taxel_affine``:  y_t ~ g_t * p_t + o_t      (2T parameters, shrunk toward the
  global affine fit with a ridge prior worth ``tau`` pseudo-readings)

Every fit is robust (least absolute deviation, the training objective): the gain is the
exact weighted median of y/p, the affine fits use IRLS. Readings of one raw (force, key)
group are calibrated or evaluated together, never split, so replicates do not leak.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from taxelnet.evaluation import _group_mean, decomposed_metrics, global_ssim

VARIANTS = ("none", "global_gain", "global_affine", "taxel_affine")
IRLS_EPS = 1.0  # TC: residual floor of the IRLS weights 1 / max(|r|, eps)
IRLS_ITERS = 100
DEFAULT_TAU = 3.0  # pseudo-readings of prior weight on the per-taxel parameters


@dataclass(frozen=True)
class Calibration:
    """y_hat = gain * p + offset; gain/offset are scalars or (T,) per-taxel arrays."""

    variant: str
    gain: np.ndarray
    offset: np.ndarray

    def apply(self, pred: np.ndarray) -> np.ndarray:
        return self.gain * pred + self.offset

    def summary(self) -> dict[str, float]:
        g, o = np.asarray(self.gain, float), np.asarray(self.offset, float)
        out = {"gain": float(np.median(g)), "offset": float(np.median(o))}
        if g.ndim:
            out |= {"gain_std": float(g.std()), "offset_std": float(o.std())}
        return out


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median: minimizer of sum_i w_i |values_i - m|."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def fit_global_gain(pred: np.ndarray, target: np.ndarray) -> Calibration:
    """Exact L1 fit of y = g p: the |p|-weighted median of y / p."""
    p, y = pred.ravel().astype(np.float64), target.ravel().astype(np.float64)
    keep = np.abs(p) > 1e-9
    g = weighted_median(y[keep] / p[keep], np.abs(p[keep])) if keep.any() else 1.0
    return Calibration("global_gain", np.array(g), np.array(0.0))


def _irls_affine(
    p: np.ndarray,
    y: np.ndarray,
    prior: tuple[np.ndarray, np.ndarray] | None = None,
    penalty: np.ndarray | None = None,
    eps: float = IRLS_EPS,
    iters: int = IRLS_ITERS,
) -> tuple[np.ndarray, np.ndarray]:
    """IRLS for min sum_i |y_it - g_t p_it - o_t| (+ ridge toward prior), per column t.

    p, y: (n, T). Returns gain, offset of shape (T,). `penalty` = (lambda_g, lambda_o)
    added to the weighted normal equations, pulling (g_t, o_t) toward `prior`.
    """
    n, t = p.shape
    g0, o0 = (np.ones(t), np.zeros(t)) if prior is None else prior
    lam = np.zeros(2) if penalty is None else penalty
    g, o = g0.copy(), o0.copy()
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - g * p - o), eps)
        a11 = (w * p * p).sum(0) + lam[0]
        a12 = (w * p).sum(0)
        a22 = w.sum(0) + lam[1]
        b1 = (w * p * y).sum(0) + lam[0] * g0
        b2 = (w * y).sum(0) + lam[1] * o0
        det = a11 * a22 - a12**2
        ok = det > 1e-12 * np.maximum(a11 * a22, 1e-300)
        g_new = np.where(ok, (a22 * b1 - a12 * b2) / np.where(ok, det, 1), g0)
        o_new = np.where(ok, (a11 * b2 - a12 * b1) / np.where(ok, det, 1), o0)
        if np.allclose(g_new, g, atol=1e-7) and np.allclose(o_new, o, atol=1e-5):
            g, o = g_new, o_new
            break
        g, o = g_new, o_new
    return g, o


def fit_global_affine(pred: np.ndarray, target: np.ndarray) -> Calibration:
    """L1 (IRLS) fit of y = g p + o with one gain and one offset for every taxel."""
    p = pred.reshape(-1, 1).astype(np.float64)
    y = target.reshape(-1, 1).astype(np.float64)
    g, o = _irls_affine(p, y)
    return Calibration("global_affine", np.array(g[0]), np.array(o[0]))


def fit_taxel_affine(
    pred: np.ndarray, target: np.ndarray, tau: float = DEFAULT_TAU
) -> Calibration:
    """Per-taxel L1 fit y_t = g_t p_t + o_t, shrunk toward the global affine fit.

    The ridge weight equals `tau` average calibration readings: lambda_g = tau *
    mean_{i,t}(w0 p^2), lambda_o = tau * mean_{i,t}(w0), with w0 the IRLS weights of the
    global fit. Quiet taxels (p ~ 0, gain unidentifiable) stay at the global gain.
    """
    p = pred.reshape(len(pred), -1).astype(np.float64)
    y = target.reshape(len(target), -1).astype(np.float64)
    glob = fit_global_affine(p, y)
    g0, o0 = float(glob.gain), float(glob.offset)
    w0 = 1.0 / np.maximum(np.abs(y - g0 * p - o0), IRLS_EPS)
    lam = tau * np.array([(w0 * p * p).mean(), w0.mean()])
    t = p.shape[1]
    g, o = _irls_affine(p, y, (np.full(t, g0), np.full(t, o0)), lam)
    return Calibration("taxel_affine", g, o)


def fit_calibration(
    variant: str, pred: np.ndarray, target: np.ndarray, tau: float = DEFAULT_TAU
) -> Calibration:
    """Fit `variant` on calibration readings pred, target (n, ...) in raw TC."""
    n = len(pred)
    p, y = pred.reshape(n, -1), target.reshape(n, -1)
    match variant:
        case "none":
            return Calibration("none", np.array(1.0), np.array(0.0))
        case "global_gain":
            return fit_global_gain(p, y)
        case "global_affine":
            return fit_global_affine(p, y)
        case "taxel_affine":
            return fit_taxel_affine(p, y, tau)
    raise ValueError(f"unknown calibration variant {variant!r}, expected {VARIANTS}")


def eval_metrics(pred: np.ndarray, target: np.ndarray, group: np.ndarray) -> dict:
    """MAE (micro, macro-group) plus the decomposed metrics, raw TC, (n, T) inputs."""
    n = len(pred)
    p = pred.reshape(n, -1).astype(np.float64)
    t = target.reshape(n, -1).astype(np.float64)
    err = np.abs(p - t).mean(1)
    m = decomposed_metrics(p, t, group)
    return {
        "micro_mae": float(err.mean()),
        "macro_mae": float(_group_mean(err, group).mean()),
        "zero_macro_mae": float(_group_mean(np.abs(t).mean(1), group).mean()),
        "cosine_median": m["cosine_median"],
        "amp_logratio_median": m["amp_logratio_median"],
        "macro_aligned_mae": m["macro_aligned_mae"],
        "macro_offset_removed_mae": m["macro_offset_removed_mae"],
        "macro_peak_hit": m["macro_peak_hit"],
        "macro_contact_iou": m["macro_contact_iou"],
    }


def _mean_std(rows: list[dict]) -> dict[str, Any]:
    keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    out: dict[str, Any] = {"n_draws": len(rows)}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
        out[f"{k}_median"] = float(np.median(vals))
    return out


def run_protocol(
    pred: np.ndarray,
    target: np.ndarray,
    group: np.ndarray,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    n_draws: int = 20,
    variants: tuple[str, ...] = VARIANTS,
    tau: float = DEFAULT_TAU,
    seed: int = 0,
    fit: Callable[..., Calibration] = fit_calibration,
) -> dict[str, Any]:
    """Few-shot calibration of one domain.

    For every k, draw k calibration groups `n_draws` times (the same draws for every
    variant), fit each variant on their readings and evaluate on the other groups.
    `k="all"` is the oracle: fit and evaluate on every reading. Returns
    {variant: {k: {metric_mean/std/median, gain_*, offset_*}}}.
    """
    units = np.unique(group)
    rng = np.random.default_rng(seed)
    draws = {
        k: [rng.choice(units, size=k, replace=False) for _ in range(n_draws)]
        for k in ks
        if k < len(units)
    }
    out: dict[str, Any] = {v: {} for v in variants}
    for v in variants:
        for k, picks in draws.items():
            rows = []
            for pick in picks:
                cal = np.isin(group, pick)
                c = fit(v, pred[cal], target[cal], tau=tau)
                r = eval_metrics(c.apply(_flat(pred[~cal])), target[~cal], group[~cal])
                rows.append(r | c.summary())
            out[v][str(k)] = _mean_std(rows)
        c = fit(v, pred, target, tau=tau)
        out[v]["all"] = eval_metrics(c.apply(_flat(pred)), target, group) | c.summary()
        if np.ndim(c.gain):
            out[v]["all"]["taxel_gain"] = np.asarray(c.gain).tolist()
            out[v]["all"]["taxel_offset"] = np.asarray(c.offset).tolist()
    out["n_readings"] = len(group)
    out["n_groups"] = len(units)
    return out


def _flat(a: np.ndarray) -> np.ndarray:
    return a.reshape(len(a), -1)


def shots_to_oracle(
    result: dict[str, Any], variant: str, tol: float = 0.10, metric: str = "macro_mae"
) -> int | None:
    """Smallest k whose mean `metric` is within `tol` (relative) of the k=all oracle."""
    res = result[variant]
    oracle = res["all"][metric]
    for k in sorted((int(k) for k in res if k != "all")):
        if res[str(k)][f"{metric}_mean"] <= (1 + tol) * oracle:
            return k
    return None


def unseen_pad_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pads: np.ndarray,
    objects: np.ndarray,
    train_scale: dict[str, float] | None,
    ssim_data_range: float,
    ks: tuple[int, ...] = (5, 20),
    n_draws: int = 50,
    seed: int = 0,
) -> dict[str, float]:
    """Test metrics for pads unseen in training, each pad rescaled from its own readings.

    `pred` is in TC with the *training* pad scale (`train_scale[pad]`, the global_scale
    std of that pad's training targets). Two ways to adapt it to a test pad, both
    computed from readings of that pad only:

    - ``testscale``: replace the training scale by the test pad's own y_scale, the std of
      its real readings (same statistic as the training scaler): pred * s_test / s_train.
      Needs only real readings. Skipped when `train_scale` is None.
    - ``padgain``: one L1 gain fitted on predicted vs real maps of the pad's readings
      (``global_gain``); needs the simulated deformation of those readings.

    `<family>_k<k>_*`: per draw, `k` random readings of each pad give its scale or gain,
    scored on every other test reading (mean over `n_draws` draws, the same draws for
    both families). `<family>_all_*` (testscale) / `padgain_oracle_*`: computed from all
    readings of the pad and scored on them (optimistic: uses every test label).
    Metrics: `_mae` (per-reading MAE averaged over readings), `_obj_mean_mae` (mean of
    the per-object MAEs), `_ssim` (mean per-reading global SSIM).
    """
    n = len(pred)
    p, t = _flat(pred).astype(np.float64), _flat(target).astype(np.float64)
    pads, objects = np.asarray(pads).astype(str), np.asarray(objects).astype(str)
    pad_rows = {q: np.flatnonzero(pads == q) for q in sorted(set(pads))}

    def factors(family: str, fit_rows: dict[str, np.ndarray]) -> dict[str, float]:
        if family == "padgain":
            return {
                q: float(fit_global_gain(p[r], t[r]).gain) for q, r in fit_rows.items()
            }
        return {q: float(t[r].std()) / train_scale[q] for q, r in fit_rows.items()}

    def score(f: dict[str, float], keep: np.ndarray) -> np.ndarray:
        cal = p.copy()
        for q, factor in f.items():
            cal[pads == q] *= factor
        err = np.abs(cal - t).mean(1)[keep]
        obj = objects[keep]
        per_obj = [err[obj == o].mean() for o in sorted(set(obj))]
        ssim = global_ssim(cal[keep], t[keep], ssim_data_range)
        return np.array([err.mean(), np.mean(per_obj), ssim.mean()])

    families = ["padgain"] + (["testscale"] if train_scale is not None else [])
    names = ("mae", "obj_mean_mae", "ssim")
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for k in ks:
        acc = {fam: [] for fam in families}
        for _ in range(n_draws):
            fit_rows = {
                q: rng.choice(r, min(k, len(r) - 1), replace=False)
                for q, r in pad_rows.items()
            }
            keep = np.ones(n, bool)
            keep[np.concatenate(list(fit_rows.values()))] = False
            for fam in families:
                acc[fam].append(score(factors(fam, fit_rows), keep))
        for fam in families:
            mean = np.mean(acc[fam], axis=0)
            out |= {f"{fam}_k{k}_{m}": float(v) for m, v in zip(names, mean)}
    everything = np.ones(n, bool)
    for fam, tag in (("padgain", "oracle"), ("testscale", "all")):
        if fam in families:
            res = score(factors(fam, pad_rows), everything)
            out |= {f"{fam}_{tag}_{m}": float(v) for m, v in zip(names, res)}
            out |= {
                f"{fam}_{tag}_factor_{q}": v for q, v in factors(fam, pad_rows).items()
            }
    return out
