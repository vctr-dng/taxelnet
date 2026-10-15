"""Audit of the merged indenter + grasp training pair against the grasp test set.

Run: PYTHONPATH=<worktree>/src python audit_merged/audit.py
Writes audit_merged/out/{summary.json, *.png}; prints a compact report.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from taxelnet.config import DataConfig
from taxelnet.dataset import load_paired, load_test_paired

D = Path("/home/vctr-dng/robotic/taxelnet/data/newton/v1")
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
CONTACT_MM = 0.05  # node counted as in contact above this depth
OFF_CONTACT_MM = 0.01  # pooled taxel depth below this = out of contact

cfg = DataConfig(
    data_dir=D / "merged",
    real_csv="Dataset_Merged_Real.csv",
    sim_csv="Dataset_Merged_Simulation.csv",
    test_dir=D / "test",
    cache_dir=D / "merged/.cache",
)
tr = load_paired(cfg)
te = load_test_paired(cfg)
S: dict = {}


def paths_of(p) -> list[str]:
    return [p.groups[g]["path"] for g in p.group]


def pool7x4(x: np.ndarray) -> np.ndarray:
    """Block-mean of (N, 77, 49) onto the 7 x 4 taxel partition."""
    rb = np.linspace(0, x.shape[1], 8).round().astype(int)
    cb = np.linspace(0, x.shape[2], 5).round().astype(int)
    return np.stack(
        [x[:, rb[i] : rb[i + 1], cb[j] : cb[j + 1]].mean((1, 2)) for i in range(7) for j in range(4)],
        1,
    )


def feats(p) -> dict[str, np.ndarray]:
    x = p.x.reshape(len(p.x), -1)
    y = p.y.reshape(len(p.y), -1)
    return {
        "xsum": np.clip(x, 0, None).sum(1),
        "xmax": x.max(1),
        "xmin": x.min(1),
        "area": (x > CONTACT_MM).sum(1),
        "ysum": np.clip(y, 0, None).sum(1),
        "ymax": y.max(1),
        "pool": pool7x4(p.x),
        "y": y,
    }


ftr, fte = feats(tr), feats(te)
tr_paths, te_names = paths_of(tr), paths_of(te)
tr_force = np.array([tr.groups[g]["force"] for g in tr.group])
te_force = np.array([te.groups[g]["force"] for g in te.group])
is_grasp = np.array([tr.groups[g]["dataset"] == "grasp" for g in tr.group])
is_ind = ~is_grasp
tr_obj = np.array([p.split("/", 1)[0] for p in tr_paths])
te_obj = np.array(["_".join(n.split("_")[:3]) for n in te_names])  # e.g. Sphere_TestG_20
gr_tokens = [p.split("/", 1)[1].split("_") for p in np.array(tr_paths)[is_grasp]]
te_tokens = [n.split("_") for n in te_names]

# --- 1. composition and coverage -------------------------------------------------------
S["composition"] = {
    "train_rows": len(tr.group),
    "indenter_rows": int(is_ind.sum()),
    "grasp_rows": int(is_grasp.sum()),
    "groups": len(tr.groups),
    "grasp_objects": dict(collections.Counter(tr_obj[is_grasp])),
    "grasp_force_label": dict(collections.Counter(tr_force[is_grasp].tolist())),
    "grasp_deg": dict(collections.Counter(t[-2] for t in gr_tokens)),
    "grasp_AB": dict(collections.Counter(t[2] for t in gr_tokens)),
    "grasp_pad": dict(collections.Counter(t[-1] for t in gr_tokens)),
    "test_deg": dict(collections.Counter(t[-2] for t in te_tokens)),
    "test_groups": dict(collections.Counter(te_obj)),
    "test_deg_unseen_in_train": sorted(
        {t[-2] for t in te_tokens} - {t[-2] for t in gr_tokens}
    ),
}

# --- 2. depth / signal ranges per source --------------------------------------------------
def rng(m, f):
    return {
        "n": int(m.sum()),
        "xmax_median_mm": round(float(np.median(f["xmax"][m])), 3),
        "xmax_p95_mm": round(float(np.quantile(f["xmax"][m], 0.95)), 3),
        "xmax_max_mm": round(float(f["xmax"][m].max()), 3),
        "xsum_median": round(float(np.median(f["xsum"][m])), 1),
        "area_median_nodes": int(np.median(f["area"][m])),
        "ysum_median_TC": round(float(np.median(f["ysum"][m])), 0),
        "ymax_max_TC": round(float(f["ymax"][m].max()), 0),
        "y_min_TC": round(float(f["y"][m].min()), 0),
    }


S["ranges"] = {
    "indenter": rng(is_ind, ftr),
    "grasp_train_20": rng(is_grasp & (tr_force == 20), ftr),
    "grasp_train_235": rng(is_grasp & (tr_force == 235), ftr),
    **{f"test_{g}": rng(te_obj == g, fte) for g in sorted(set(te_obj))},
}

# --- 3. force law from the indenter data and implied forces -------------------------------
a, b = np.polyfit(tr_force[is_ind], ftr["xsum"][is_ind], 1)
r = np.corrcoef(tr_force[is_ind], ftr["xsum"][is_ind])[0, 1]
implied = lambda xs: (xs - b) / a  # noqa: E731
S["force_law"] = {"xsum_per_N": round(a, 3), "intercept": round(b, 2), "r": round(r, 4)}
imp = {}
for lab in (20.0, 235.0):
    m = is_grasp & (tr_force == lab)
    q = np.quantile(implied(ftr["xsum"][m]), [0.1, 0.5, 0.9])
    imp[f"grasp_train_label{int(lab)}"] = [round(v, 1) for v in q]
for g in sorted(set(te_obj)):
    q = np.quantile(implied(fte["xsum"][te_obj == g]), [0.1, 0.5, 0.9])
    imp[f"test_{g}"] = [round(v, 1) for v in q]
S["implied_force_N_p10_p50_p90"] = imp

# --- 4. input->signal law: indenter fit, ratio real / expected ---------------------------
mi = is_ind & (ftr["xsum"] > 1) & (ftr["ysum"] > 100)
slope, inter = np.polyfit(np.log(ftr["xsum"][mi]), np.log(ftr["ysum"][mi]), 1)
expected = lambda xs: np.exp(inter) * np.clip(xs, 1e-3, None) ** slope  # noqa: E731
S["signal_law_indenter"] = {"loglog_slope": round(slope, 3)}


def ratio(f, m):
    rr = f["ysum"][m] / expected(f["xsum"][m])
    return [round(float(v), 3) for v in np.quantile(rr, [0.25, 0.5, 0.75])]


ratios = {"indenter (self)": ratio(ftr, is_ind)}
for o in sorted(set(tr_obj[is_grasp])):
    ratios[f"train {o}"] = ratio(ftr, is_grasp & (tr_obj == o))
for pad in ("S1", "S2"):
    m = is_grasp & np.array([p.endswith(pad) for p in tr_paths])
    ratios[f"train grasp {pad}"] = ratio(ftr, m)
for lab in (20.0, 235.0):
    ratios[f"train grasp label{int(lab)}"] = ratio(ftr, is_grasp & (tr_force == lab))
for g in sorted(set(te_obj)):
    ratios[f"test {g}"] = ratio(fte, te_obj == g)
S["real_over_indenter_law_q25_q50_q75"] = ratios

# grasp-only law (train grasp), applied to the test groups
mg = is_grasp & (ftr["xsum"] > 1) & (ftr["ysum"] > 100)
gs, gi = np.polyfit(np.log(ftr["xsum"][mg]), np.log(ftr["ysum"][mg]), 1)
S["signal_law_train_grasp"] = {
    "loglog_slope": round(gs, 3),
    "corr_log": round(float(np.corrcoef(np.log(ftr["xsum"][mg]), np.log(ftr["ysum"][mg]))[0, 1]), 3),
}
S["test_real_over_grasp_law_q25_q50_q75"] = {
    g: [
        round(float(v), 3)
        for v in np.quantile(
            fte["ysum"][te_obj == g] / (np.exp(gi) * fte["xsum"][te_obj == g] ** gs), [0.25, 0.5, 0.75]
        )
    ]
    for g in sorted(set(te_obj))
}

# --- 5. S1 vs S2 of the same grasp -----------------------------------------------------
gidx = np.where(is_grasp)[0]
pair_key = {}
for i in gidx:
    pair_key.setdefault(tr_paths[i].rsplit("_", 1)[0], {})[tr_paths[i][-2:]] = i
pairs = [(v["S1"], v["S2"]) for v in pair_key.values() if "S1" in v and "S2" in v]
p1, p2 = np.array(pairs).T
same_sim = np.mean([np.allclose(tr.x[i], tr.x[j]) for i, j in pairs])
mirror_sim = np.mean([np.allclose(tr.x[i], tr.x[j][:, ::-1], atol=1e-3) for i, j in pairs[:500]])
S["pads"] = {
    "pairs": len(pairs),
    "frac_identical_sim": round(float(same_sim), 3),
    "frac_mirror_sim_first500": round(float(mirror_sim), 3),
    "median_ysum_S1": round(float(np.median(ftr["ysum"][p1])), 0),
    "median_ysum_S2": round(float(np.median(ftr["ysum"][p2])), 0),
    "median_xsum_S1": round(float(np.median(ftr["xsum"][p1])), 1),
    "median_xsum_S2": round(float(np.median(ftr["xsum"][p2])), 1),
    "median_log_ysum_ratio_S2_over_S1": round(
        float(np.median(np.log((ftr["ysum"][p2] + 1) / (ftr["ysum"][p1] + 1)))), 3
    ),
}

# --- 6. pattern agreement: pooled sim vs real ------------------------------------------
def cos(a_, b_):
    return (a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1) + 1e-9)


S["cosine_pooled_sim_vs_real_median"] = {
    "indenter": round(float(np.median(cos(ftr["pool"][is_ind], ftr["y"][is_ind]))), 3),
    **{
        f"train {o}": round(float(np.median(cos(ftr["pool"][tr_obj == o], ftr["y"][tr_obj == o]))), 3)
        for o in sorted(set(tr_obj[is_grasp]))
    },
    **{
        f"test {g}": round(float(np.median(cos(fte["pool"][te_obj == g], fte["y"][te_obj == g]))), 3)
        for g in sorted(set(te_obj))
    },
}

# --- 7. off-contact offsets -------------------------------------------------------------
def off_contact(f, m):
    off = f["pool"][m] < OFF_CONTACT_MM
    vals = f["y"][m][off]
    return {
        "median_TC": round(float(np.median(vals)), 1),
        "frac_below_-50": round(float((vals < -50).mean()), 3),
        "min_TC": round(float(vals.min()), 0),
    }


S["off_contact_taxels"] = {
    "indenter": off_contact(ftr, is_ind),
    "grasp_train": off_contact(ftr, is_grasp),
    **{f"test {g}": off_contact(fte, te_obj == g) for g in sorted(set(te_obj))},
}

# --- 8. nearest neighbours of test rows in training (pooled sim, 28-d) ------------------
P = ftr["pool"]
nn_rows = []
for g in sorted(set(te_obj)):
    idx = np.where(te_obj == g)[0]
    d = ((fte["pool"][idx][:, None, :] - P[None]) ** 2).sum(-1)
    k = np.argsort(d, 1)[:, :5]
    src_grasp = is_grasp[k].mean()
    yr = fte["ysum"][idx] / (ftr["ysum"][k].mean(1) + 1)
    nn_rows.append(
        {
            "group": g,
            "nn5_from_grasp_frac": round(float(src_grasp), 3),
            "nn1_dist_median": round(float(np.sqrt(d.min(1)).mean()), 4),
            "real_sum_over_nn5_median": round(float(np.median(yr)), 3),
            "nn_objects": dict(collections.Counter(tr_obj[k[:, 0]]).most_common(3)),
        }
    )
# reference: within-train leave-object-out NN distance for grasp objects
ref = []
for o in sorted(set(tr_obj[is_grasp])):
    m = tr_obj == o
    sub = np.where(m)[0][:200]
    d = ((P[sub][:, None, :] - P[~m][None]) ** 2).sum(-1)
    ref.append(np.sqrt(d.min(1)).mean())
S["test_nearest_train_neighbours"] = nn_rows
S["train_grasp_leave_object_out_nn1_dist_mean"] = round(float(np.mean(ref)), 4)

# --- 9. std columns of the grasp files ---------------------------------------------------
std_cols = [f"std{i}" for i in range(1, 29)]
g_std = pl.read_csv(D / "training/Dataset_Grasp_Real_qtac100.csv", columns=std_cols).to_numpy()
t_std = pl.read_csv(D / "test/Dataset_Grasp_Real_qtac100_testing.csv", columns=std_cols).to_numpy()
S["std_columns"] = {
    "train_grasp_median": round(float(np.median(g_std)), 2),
    "train_grasp_p95": round(float(np.quantile(g_std, 0.95)), 2),
    "test_median": round(float(np.median(t_std)), 2),
    "corr_std_absy_train": round(float(np.corrcoef(g_std.ravel(), np.abs(ftr["y"][is_grasp]).ravel())[0, 1]), 3),
}

# --- 10. duplicates ----------------------------------------------------------------------
gx = tr.x[is_grasp].reshape(int(is_grasp.sum()), -1)
_, inv, cnt = np.unique(np.round(gx, 4), axis=0, return_inverse=True, return_counts=True)
S["duplicates"] = {
    "grasp_rows_with_identical_sim_to_another_row": int((cnt[inv] > 1).sum()),
    "n_sim_differ_keys": tr.n_sim_differ,
}

(OUT / "summary.json").write_text(json.dumps(S, indent=1, default=str))

# --- figures -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
xs = np.linspace(1, max(ftr["xsum"].max(), fte["xsum"].max()), 200)
ax[0].scatter(ftr["xsum"][is_ind], ftr["ysum"][is_ind], s=2, alpha=0.15, c="0.6", label="indenter")
for lab, c in ((20.0, "tab:blue"), (235.0, "tab:orange")):
    m = is_grasp & (tr_force == lab)
    ax[0].scatter(ftr["xsum"][m], ftr["ysum"][m], s=3, alpha=0.3, c=c, label=f"train grasp {int(lab)}")
for g, mk in zip(sorted(set(te_obj)), "xs^D"):
    m = te_obj == g
    ax[0].scatter(fte["xsum"][m], fte["ysum"][m], s=14, marker=mk, c="k", label=f"test {g}")
ax[0].plot(xs, expected(xs), "r-", lw=1, label=f"indenter law (slope {slope:.2f})")
ax[0].plot(xs, np.exp(gi) * xs**gs, "g--", lw=1, label=f"train-grasp law (slope {gs:.2f})")
ax[0].set(xlabel="Σ positive sim deformation (mm)", ylabel="Σ positive real signal (TC)",
          xscale="log", yscale="log", title="input→signal law")
ax[0].legend(fontsize=7, markerscale=2)
objs = sorted(set(tr_obj[is_grasp]))
med = [ratios[f"train {o}"][1] for o in objs] + [ratios[f"test {g}"][1] for g in sorted(set(te_obj))]
lbl = [f"train {o}" for o in objs] + [f"TEST {g}" for g in sorted(set(te_obj))]
ax[1].barh(range(len(med)), med, color=["tab:blue"] * len(objs) + ["k"] * len(set(te_obj)))
ax[1].axvline(1, c="r", lw=1)
ax[1].set_yticks(range(len(med)), lbl, fontsize=7)
ax[1].set(xlabel="median real Σ / indenter-law expectation", title="gain vs indenter data")
fig.tight_layout()
fig.savefig(OUT / "fig_law_merged.png", dpi=110)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0, max(ftr["xmax"].max(), fte["xmax"].max()), 60)
ax[0].hist(ftr["xmax"][is_ind], bins, alpha=0.5, density=True, label="indenter")
ax[0].hist(ftr["xmax"][is_grasp], bins, alpha=0.5, density=True, label="train grasp")
ax[0].hist(fte["xmax"], bins, alpha=0.5, density=True, label="test grasp")
ax[0].set(xlabel="max sim depth (mm)", title="depth coverage")
ax[0].legend()
ax[1].scatter(ftr["ysum"][p1], ftr["ysum"][p2], s=3, alpha=0.3)
lim = max(ftr["ysum"][p1].max(), ftr["ysum"][p2].max())
ax[1].plot([0, lim], [0, lim], "r-", lw=1)
ax[1].set(xlabel="Σ real S1 (TC)", ylabel="Σ real S2 (TC)", title="pads of the same grasp")
fig.tight_layout()
fig.savefig(OUT / "fig_coverage_pads.png", dpi=110)
plt.close(fig)

print(json.dumps(S, indent=1, default=str))
