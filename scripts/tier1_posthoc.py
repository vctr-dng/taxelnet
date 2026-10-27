"""Post-hoc (no retraining) test-time analyses on run bundles, for test pads unseen in training.

Reads each bundle's predictions_test.parquet (raw TC, training-pad scale). Every
calibration is fitted per test pad (S1, S2) on k random readings of that pad and scored
on all other test readings, with the same draws for every method; `oracle` fits on all
readings of the pad and scores on them (optimistic). Methods:

- ``gain``:   y = g p                         (one L1 gain per pad, the reference)
- ``power``:  y = g sign(p) |p|^gamma          (gamma on a grid, L1-optimal on the fit
  readings; corrects amplitude compression)
- ``taxaff_tau<t>``: per-taxel gain + offset (calibration.fit_taxel_affine, prior `t`)
- ``row_tau<t>`` / ``taxel_tau<t>``: one gain per taxel row (7) / per taxel (28), each an
  L1 weighted median of y/p shrunk toward the pad gain by `t` pseudo-readings
- ``shift``: predicted maps translated by (drow, dcol) in {-1, 0, 1}^2 (vacated cells 0)
  before the oracle pad gain; a registration check per object
- ``ensemble``: per arm, predictions averaged over the arm's bundles, then ``gain``

Usage: python scripts/tier1_posthoc.py --arm NAME BUNDLE... [--arm NAME BUNDLE...] --out DIR
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from taxelnet.calibration import fit_calibration, fit_global_gain, weighted_median
from taxelnet.evaluation import object_group

GRID = (7, 4)
KS = (5, 10, 20)
TAUS = (0.0, 1.0, 3.0, 10.0, 30.0)
GAMMAS = np.round(np.arange(0.5, 1.81, 0.05), 2)

Fit = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def load_predictions(bundle: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pl.read_parquet(bundle / "predictions_test.parquet").sort("source_row")
    pred = np.stack(df["prediction"].to_list()).astype(np.float64)
    target = np.stack(df["target"].to_list()).astype(np.float64)
    return pred, target, df["path"].to_list()


def fit_gain(p: np.ndarray, y: np.ndarray):
    g = float(fit_global_gain(p, y).gain)
    return lambda q: g * q


def fit_power(p: np.ndarray, y: np.ndarray):
    best = None
    for gamma in GAMMAS:
        z = np.sign(p) * np.abs(p) ** gamma
        g = float(fit_global_gain(z, y).gain)
        loss = np.abs(y - g * z).sum()
        if best is None or loss < best[0]:
            best = (loss, gamma, g)
    _, gamma, g = best
    return lambda q: g * np.sign(q) * np.abs(q) ** gamma


def shrunk_gains(p: np.ndarray, y: np.ndarray, cells: np.ndarray, tau: float):
    """One L1 gain per cell group (row or taxel), shrunk toward the pad gain g0.

    The prior is a pseudo-observation y/p = g0 whose weight is `tau` times the per-reading
    |p| mass of an average cell group of the pad (the same for every group, like the
    ridge prior of calibration.fit_taxel_affine): rarely activated taxels stay near g0,
    only strongly activated ones move. `tau` counts average-cell pseudo-readings.
    """
    g0 = float(fit_global_gain(p, y).gain)
    n_cells = cells.max() + 1
    gains = np.full(n_cells, g0)
    prior_w = tau * np.abs(p).sum(1).mean() / n_cells
    for c in range(n_cells):
        sel = cells == c
        pc, yc = p[:, sel].ravel(), y[:, sel].ravel()
        keep = np.abs(pc) > 1e-9
        values = np.append(yc[keep] / pc[keep], g0)
        weights = np.append(np.abs(pc[keep]), prior_w)
        if weights.sum() > 0:
            gains[c] = weighted_median(values, weights)
    return lambda q: q * gains[cells]


def fit_taxel_affine(tau: float) -> Fit:
    """Per-taxel gain + offset (calibration.fit_taxel_affine, IRLS, ridge prior `tau`)."""

    def fit(p: np.ndarray, y: np.ndarray):
        return fit_calibration("taxel_affine", p, y, tau).apply

    return fit


def fit_rows(tau: float) -> Fit:
    cells = np.repeat(np.arange(GRID[0]), GRID[1])
    return lambda p, y: shrunk_gains(p, y, cells, tau)


def fit_taxels(tau: float) -> Fit:
    cells = np.arange(GRID[0] * GRID[1])
    return lambda p, y: shrunk_gains(p, y, cells, tau)


def shifted(pred: np.ndarray, dr: int, dc: int) -> np.ndarray:
    m = pred.reshape(len(pred), *GRID)
    out = np.zeros_like(m)
    rs, rd = (
        (slice(0, GRID[0] - dr), slice(dr, None))
        if dr >= 0
        else (slice(-dr, None), slice(0, GRID[0] + dr))
    )
    cs, cd = (
        (slice(0, GRID[1] - dc), slice(dc, None))
        if dc >= 0
        else (slice(-dc, None), slice(0, GRID[1] + dc))
    )
    out[:, rd, cd] = m[:, rs, cs]
    return out.reshape(len(pred), -1)


class Scorer:
    """Shared random draws of k calibration readings per pad; MAE overall and per object."""

    def __init__(self, pads: np.ndarray, objects: np.ndarray, n_draws: int, seed: int):
        self.pads, self.objects = pads, objects
        self.pad_rows = {q: np.flatnonzero(pads == q) for q in sorted(set(pads))}
        rng = np.random.default_rng(seed)
        self.draws = {
            k: [{q: rng.choice(r, k, replace=False) for q, r in self.pad_rows.items()}
                for _ in range(n_draws)]
            for k in KS
        }  # fmt: skip

    def _apply(self, pred, target, fit: Fit, fit_rows) -> np.ndarray:
        out = np.empty_like(pred)
        for q, rows in fit_rows.items():
            sel = self.pads == q
            out[sel] = fit(pred[rows], target[rows])(pred[sel])
        return out

    def _metrics(self, cal, target, keep) -> dict[str, float]:
        err = np.abs(cal - target).mean(1)
        obj = self.objects[keep]
        per = {o: float(err[keep][obj == o].mean()) for o in sorted(set(obj))}
        return {
            "mae": float(err[keep].mean()),
            "obj_mean_mae": float(np.mean(list(per.values()))),
        } | {f"obj_{o}_mae": v for o, v in per.items()}

    def score(self, pred, target, fit: Fit) -> dict[str, dict[str, float]]:
        res = {}
        for k, draws in self.draws.items():
            rows = []
            for fit_rows in draws:
                keep = np.ones(len(pred), bool)
                keep[np.concatenate(list(fit_rows.values()))] = False
                rows.append(
                    self._metrics(
                        self._apply(pred, target, fit, fit_rows), target, keep
                    )
                )
            res[f"k{k}"] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
        all_rows = np.ones(len(pred), bool)
        res["oracle"] = self._metrics(
            self._apply(pred, target, fit, self.pad_rows), target, all_rows
        )
        return res


def mean_over(results: list[dict]) -> dict:
    """Element-wise mean (and std) over bundles of nested {method: {k: {metric: x}}}."""
    out: dict = {}
    for method in results[0]:
        out[method] = {}
        for k in results[0][method]:
            out[method][k] = {}
            for m in results[0][method][k]:
                v = np.array([r[method][k][m] for r in results])
                out[method][k][m] = float(v.mean())
                out[method][k][f"{m}_std"] = float(v.std())
    return out


def analyse_arm(bundles: list[Path], n_draws: int, seed: int) -> dict:
    preds, targets, names = zip(*(load_predictions(b) for b in bundles))
    target, names = targets[0], names[0]
    pads = np.array([n.rsplit("_", 1)[1] for n in names])
    objects = np.array([object_group(n) for n in names])
    scorer = Scorer(pads, objects, n_draws, seed)
    methods: dict[str, Fit] = {"gain": fit_gain, "power": fit_power}
    methods |= {f"row_tau{t:g}": fit_rows(t) for t in TAUS}
    methods |= {f"taxel_tau{t:g}": fit_taxels(t) for t in TAUS}
    methods |= {f"taxaff_tau{t:g}": fit_taxel_affine(t) for t in (1.0, 3.0, 10.0)}
    per_bundle = []
    for pred in preds:
        r = {name: scorer.score(pred, target, fit) for name, fit in methods.items()}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                s = scorer._metrics(
                    scorer._apply(shifted(pred, dr, dc), target, fit_gain, scorer.pad_rows),
                    target, np.ones(len(pred), bool),
                )  # fmt: skip
                r[f"shift_r{dr:+d}_c{dc:+d}"] = {"oracle": s}
        per_bundle.append(r)
    out = {"n_bundles": len(bundles), "single": mean_over(per_bundle)}
    if len(bundles) > 1:
        ens = np.mean(preds, axis=0)
        out["ensemble"] = {"gain": scorer.score(ens, target, fit_gain)}
        out["ensemble"]["power"] = scorer.score(ens, target, fit_power)
        out["ensemble"]["taxaff_tau3"] = scorer.score(
            ens, target, fit_taxel_affine(3.0)
        )
    return out


def report(results: dict, out_dir: Path) -> None:
    lines = []
    for arm, res in results.items():
        s = res["single"]
        objs = sorted(
            k[4:-4]
            for k in s["gain"]["oracle"]
            if k.startswith("obj_") and k != "obj_mean_mae" and not k.endswith("_std")
        )
        lines += [f"## {arm} ({res['n_bundles']} bundles; single models, mean ± std)", "",
                  "| method | k5 | k10 | k20 | oracle | k5 4-obj mean |", "|---|---|---|---|---|---|"]  # fmt: skip
        for m, r in s.items():
            if m.startswith("shift"):
                continue
            cells = [
                f"{r[k]['mae']:.1f} ± {r[k]['mae_std']:.1f}"
                for k in ("k5", "k10", "k20", "oracle")
            ]
            lines.append(
                f"| {m} | " + " | ".join(cells) + f" | {r['k5']['obj_mean_mae']:.1f} |"
            )
        if "ensemble" in res:
            for m, r in res["ensemble"].items():
                cells = [f"{r[k]['mae']:.1f}" for k in ("k5", "k10", "k20", "oracle")]
                lines.append(
                    f"| **ensemble {m}** | "
                    + " | ".join(cells)
                    + f" | {r['k5']['obj_mean_mae']:.1f} |"
                )
        lines += ["", "Shift (oracle pad gain), per object MAE:", "",
                  "| shift (row, col) | all | " + " | ".join(objs) + " |",
                  "|---|---|" + "---|" * len(objs)]  # fmt: skip
        for m, r in s.items():
            if m.startswith("shift"):
                o = r["oracle"]
                lines.append(
                    f"| {m[6:]} | {o['mae']:.1f} | "
                    + " | ".join(f"{o[f'obj_{x}_mae']:.1f}" for x in objs)
                    + " |"
                )
        lines.append("")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm", nargs="+", action="append", required=True, metavar="NAME BUNDLE"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, *bundles in args.arm:
        results[name] = analyse_arm([Path(b) for b in bundles], args.draws, args.seed)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    report(results, args.out)


if __name__ == "__main__":
    main()
