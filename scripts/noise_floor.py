"""Noise-floor estimate on the grasp training rows from near-duplicate inputs.

Usage: python scripts/noise_floor.py [--out outputs/capacity/noise_floor.json]
For each row, find its nearest neighbour in PCA space of the (max-abs scaled) simulated
Z grid among rows of the same object and pad (excluding itself), and take the L1 distance
between their real maps (per-taxel mean, TC). Reported in input-distance buckets, per pad,
and for leave-one-object-out neighbours (same pad, other objects). If the real map is
y = f(x) + noise with iid noise, E|y_i - y_j| ~ 1.13 sigma_L1*sqrt(2)/... : for Gaussian
noise MAE_floor = E|e| and E|e_i - e_j| = sqrt(2) E|e|, so floor ~ nn_diff / sqrt(2) when
the neighbours are true duplicates (f(x_i) = f(x_j)); otherwise it is an upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from taxelnet.config import Configs
from taxelnet.dataset import load_paired

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path("outputs/capacity/noise_floor.json"))
ap.add_argument("--n-pca", type=int, default=30)
a = ap.parse_args()

c = Configs()
c.data.data_dir, c.data.real_csv = Path("data_sensor/all"), "Dataset_GraspOnly_Real.csv"
c.data.sim_csv, c.data.cache_dir = "Dataset_GraspOnly_Simulation.csv", Path("data_sensor/.cache")
p = load_paired(c.data)
n = len(p.group)
x = p.x.reshape(n, -1).astype(np.float64)
y = p.y.reshape(n, -1).astype(np.float64)
path = np.array([p.groups[g]["path"] for g in p.group])
obj = np.array([s.split("/")[0] for s in path])
pad = np.array([s[-2:] for s in path])
force = np.array([p.groups[g]["force"] for g in p.group])
x = x / np.abs(x).max()
xc = x - x.mean(0)
u, s, vt = np.linalg.svd(xc, full_matrices=False)
z = xc @ vt[: a.n_pca].T
print("PCA var explained", float((s[: a.n_pca] ** 2).sum() / (s**2).sum()))
norm = np.linalg.norm(x, axis=1)

d = np.linalg.norm(z[:, None] - z[None], axis=2)
np.fill_diagonal(d, np.inf)
rel = lambda i, j: d[i, j] / (0.5 * (norm[i] + norm[j]) + 1e-12)
l1 = lambda i, j: np.abs(y[i] - y[j]).mean(1)
idx = np.arange(n)

out = {"n": n, "zero_pred_mae": float(np.abs(y).mean()), "pca_var": float((s[: a.n_pca] ** 2).sum() / (s**2).sum())}
for name, mask in {
    "same_object_same_pad": (obj[:, None] == obj[None]) & (pad[:, None] == pad[None]),
    "other_object_same_pad": (obj[:, None] != obj[None]) & (pad[:, None] == pad[None]),
    "same_object_other_pad": (obj[:, None] == obj[None]) & (pad[:, None] != pad[None]),
}.items():
    dm = np.where(mask, d, np.inf)
    nn = dm.argmin(1)
    r, e = rel(idx, nn), l1(idx, nn)
    res = {"nn_target_l1_mean": float(e.mean()), "nn_target_l1_median": float(np.median(e)),
           "rel_dist_median": float(np.median(r)), "by_pad": {q: float(e[pad == q].mean()) for q in ("S1", "S2")},
           "buckets": {}}
    qs = np.quantile(r, [0, .1, .25, .5, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        sel = (r >= lo) & (r < hi if hi < qs[-1] else r <= hi)
        res["buckets"][f"rel_dist {lo:.3f}-{hi:.3f}"] = {"n": int(sel.sum()), "l1": float(e[sel].mean()), "l1_S1": float(e[sel & (pad == 'S1')].mean()), "l1_S2": float(e[sel & (pad == 'S2')].mean())}
    out[name] = res
    print(name, json.dumps(res, indent=1))
# paired sim rows: S1 vs S2 sim at identical pose
a.out.write_text(json.dumps(out, indent=1))
