"""Quantify how much the leaky protocol overstates accuracy, from the saved best-val models.

For every split JSON in data/splits and every checkpoint outputs/pushing_split<S>_seed<M>_best.pt:
  test_clean_multi: test_clean readings whose (force, path) key has >= 2 real readings
  test_leaky:       readings sharing their key with train rows
Per-reading errors are averaged over the training seeds of a split, then groups are resampled
(bootstrap over (force, path) keys) to get 95% intervals. The noise floor of a reading is its MAE
against the per-taxel median of the *other* readings of the same key (leave-one-out).

Usage: uv run python scripts/analyze_pushing_leakage.py [--boot 4000]
"""

import argparse
import json
import re
from pathlib import Path

import mlflow
import numpy as np
import torch

from taxelnet.evaluation import global_ssim
from train_pushing_baseline import CONFIG, SSIM_DATA_RANGE, TAXEL_SHAPE, PushingCNN, load_and_align


def loo_median_mae(y: np.ndarray, group: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Per-reading MAE vs the median of the other readings of its key (any partition)."""
    out = np.empty(len(rows))
    members: dict[int, np.ndarray] = {}
    for k, r in enumerate(rows):
        g = group[r]
        if g not in members:
            members[g] = np.flatnonzero(group == g)
        others = members[g][members[g] != r]
        out[k] = np.abs(y[r] - np.median(y[others], axis=0)).mean()
    return out


def group_sums(values: np.ndarray, g: np.ndarray):
    uniq, inv = np.unique(g, return_inverse=True)
    return np.bincount(inv, weights=values, minlength=len(uniq)), np.bincount(inv, minlength=len(uniq))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--experiment", default="pushing-leakage-v3")
    args = ap.parse_args()

    X, y, keys = load_and_align(args.raw_dir)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    group = np.array([uniq[k] for k in keys])
    n_per_group = np.bincount(group)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    per_split = {}
    for ckpt_dir in sorted(Path("outputs").glob("pushing_split*_seed*_best.pt")):
        s, m = map(int, re.match(r"pushing_split(\d+)_seed(\d+)_best", ckpt_dir.stem).groups())
        per_split.setdefault(s, {})[m] = ckpt_dir

    results = {}
    boots = {}
    for s, ckpts in sorted(per_split.items()):
        split = json.loads(Path(f"data/splits/pushing_split_s{s}.json").read_text())
        rows = {p: np.array(r) for p, r in split["rows"].items()}
        rows["test_clean_multi"] = rows["test_clean"][n_per_group[group[rows["test_clean"]]] >= 2]
        sets = {"clean": rows["test_clean_multi"], "leaky": rows["test_leaky"]}

        mae = {k: [] for k in sets}
        ssim = {k: [] for k in sets}
        per_seed = {}
        for m, path in sorted(ckpts.items()):
            ck = torch.load(path, map_location=dev)
            model = PushingCNN(CONFIG).to(dev)
            model.load_state_dict(ck["state"])
            model.eval()
            for k, r in sets.items():
                xt = torch.from_numpy(X[r] / ck["x_max"]).to(dev)
                with torch.no_grad():
                    pred = model(xt).cpu().numpy()
                mae[k].append(np.abs(pred - y[r]).mean(1))
                ssim[k].append(
                    global_ssim(pred.reshape(-1, *TAXEL_SHAPE), y[r].reshape(-1, *TAXEL_SHAPE), SSIM_DATA_RANGE)
                )
            per_seed[m] = {k: float(mae[k][-1].mean()) for k in sets}
        vals = {}
        for k, r in sets.items():
            vals[f"mae_{k}"] = np.mean(mae[k], axis=0)  # seed-averaged per-reading error
            vals[f"ssim_{k}"] = np.mean(ssim[k], axis=0)
            vals[f"floor_{k}"] = loo_median_mae(y, group, r)
        # bootstrap over keys, independently per set (the two sets have disjoint readings)
        draws = {}
        for k, r in sets.items():
            g = group[r]
            n_groups = len(np.unique(g))
            # row 0 is the unresampled data (point estimate), the rest are bootstrap draws
            pick = np.vstack([np.arange(n_groups), rng.integers(0, n_groups, size=(args.boot, n_groups))])
            for name in ("mae", "ssim", "floor"):
                gs, gc = group_sums(vals[f"{name}_{k}"], g)
                draws[f"{name}_{k}"] = gs[pick].sum(1) / gc[pick].sum(1)
        boots[s] = draws
        results[s] = {
            "n_seeds": len(ckpts),
            "per_seed_mae": per_seed,
            "n_rows": {k: len(r) for k, r in sets.items()},
            "n_groups": {k: int(len(np.unique(group[r]))) for k, r in sets.items()},
        }

    def stats(d: dict) -> dict:
        return {
            "mae_clean": d["mae_clean"],
            "mae_leaky": d["mae_leaky"],
            "gap_mae": d["mae_clean"] - d["mae_leaky"],
            "ratio_clean_over_leaky": d["mae_clean"] / d["mae_leaky"],
            "reduction_pct": 100 * (1 - d["mae_leaky"] / d["mae_clean"]),
            "gap_ssim": d["ssim_leaky"] - d["ssim_clean"],
            "floor_clean": d["floor_clean"],
            "floor_leaky": d["floor_leaky"],
            "leaky_over_floor": d["mae_leaky"] / d["floor_leaky"],
            "clean_over_floor": d["mae_clean"] / d["floor_clean"],
            "gap_share_of_excess": (d["mae_clean"] - d["mae_leaky"]) / (d["mae_clean"] - d["floor_clean"]),
        }

    def summarize(draws: dict) -> dict:
        return {k: [float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5))] for k, v in stats(draws).items()}

    for s in boots:
        results[s]["bootstrap"] = summarize(boots[s])
    pooled = {k: np.mean([boots[s][k] for s in boots], axis=0) for k in next(iter(boots.values()))}
    results["pooled"] = {"bootstrap": summarize(pooled), "n_splits": len(boots)}

    # split-to-split and seed-to-seed spread of the point estimates
    gaps_split = [results[s]["bootstrap"]["gap_mae"][0] for s in boots]
    results["pooled"]["gap_mae_split_std"] = float(np.std(gaps_split, ddof=1)) if len(gaps_split) > 1 else None
    seed_gaps = [v["clean"] - v["leaky"] for s in boots for v in results[s]["per_seed_mae"].values()]
    results["pooled"]["gap_mae_per_run"] = [round(g, 2) for g in seed_gaps]
    results["pooled"]["gap_mae_per_run_std"] = float(np.std(seed_gaps, ddof=1)) if len(seed_gaps) > 1 else None

    out = Path("outputs/leakage_analysis.json")
    out.write_text(json.dumps(results, indent=1))
    show = ["mae_clean", "mae_leaky", "gap_mae", "reduction_pct", "ratio_clean_over_leaky", "gap_ssim",
            "floor_clean", "floor_leaky", "leaky_over_floor", "clean_over_floor", "gap_share_of_excess"]
    cols = [(f"split {s}", results[s]["bootstrap"]) for s in boots] + [("pooled", results["pooled"]["bootstrap"])]
    print(f"{'':24}" + "".join(f"{c[0]:>26}" for c in cols))
    for k in show:
        print(f"{k:24}" + "".join(f"{v[k][0]:>10.3f} [{v[k][1]:6.3f},{v[k][2]:7.3f}]" for _, v in cols))
    print("seeds per split:", {s: results[s]["n_seeds"] for s in boots})
    print("gap per run:", results["pooled"]["gap_mae_per_run"], "std", results["pooled"]["gap_mae_per_run_std"])
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="leakage-analysis"):
        mlflow.log_params({"boot": args.boot, "n_splits": len(boots), "n_runs": len(seed_gaps)})
        mlflow.log_metrics({f"pooled_{k}": v[0] for k, v in results["pooled"]["bootstrap"].items()})
        mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
