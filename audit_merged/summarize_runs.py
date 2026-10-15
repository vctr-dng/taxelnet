"""Collect merged-data baseline metrics (+ indenter-only Phase-0 references) into one table.

Also recomputes val MAE per source (indenter vs grasp rows) from each bundle's model.
Writes audit_merged/out/baseline_runs.{json,md}.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from taxelnet.checkpoint import load_model
from taxelnet.config import DataConfig
from taxelnet.dataset import TaxelDataset, load_paired
from taxelnet.evaluation import predict
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import make_split

WT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "out"
D = Path("/home/vctr-dng/robotic/taxelnet/data/newton/v1")
MERGED = sorted((WT / "outputs/merged_baseline").glob("*_m0-*"))
PHASE0 = sorted(Path("/home/vctr-dng/robotic/taxelnet/outputs/lab/phase0").glob("*_p0-*"))
CALIB = {
    p.parent.name.split("_", 1)[1]: p
    for root in (WT / "outputs/merged_baseline/calib", Path("/home/vctr-dng/robotic/taxelnet/outputs/lab/calib"))
    for p in root.glob("*/calibration.json")
}
GROUPS = ["Cuboid_TestXS_20", "Sphere_TestG_20", "Cylinder_TestM_235", "Cylinder_TestButtons_235"]


def calib_k5(run: str) -> float | None:
    p = CALIB.get(run)
    if p is None:
        return None
    c = json.loads(p.read_text())
    return c.get("test_object_macro_mae", {}).get("global_gain", {}).get("5")


def per_source_val(bundle: Path) -> dict[str, float]:
    cfg_all = json.loads((bundle / "config.json").read_text())
    dcfg = cfg_all["data"]
    data = DataConfig(
        data_dir=Path(dcfg["data_dir"]),
        real_csv=dcfg["real_csv"],
        sim_csv=dcfg["sim_csv"],
        key_cols=tuple(dcfg["key_cols"]),
        val_ratio=dcfg["val_ratio"],
        split_seed=dcfg["split_seed"],
        split_strategy=dcfg.get("split_strategy", "group"),
        cache_dir=Path(dcfg["cache_dir"]) if dcfg["cache_dir"] else None,
    )
    paired = load_paired(data)
    rows = make_split(paired, data).rows["val"]
    pre = Preprocessor.load(bundle / "preprocessing.json")
    model = load_model(bundle / "model.pt")
    pred = predict(model, TaxelDataset(paired, rows, pre.transform_x, pre.transform_y), pre)
    err = np.abs(pred - paired.y[rows]).reshape(len(rows), -1).mean(1)
    grasp = np.array([paired.groups[g]["dataset"] == "grasp" for g in paired.group[rows]])
    return {
        "val_indenter_micro_mae": float(err[~grasp].mean()),
        "val_grasp_micro_mae": float(err[grasp].mean()) if grasp.any() else float("nan"),
        "val_grasp_rows": int(grasp.sum()),
    }


def row(bundle: Path, train_data: str, source_split: bool) -> dict:
    m = json.loads((bundle / "metrics.json").read_text())
    v, t = m["val"], m["test"]
    run = bundle.name.split("_", 1)[1]
    r = {
        "run": run,
        "train_data": train_data,
        "best_epoch": m["best_epoch"],
        "val_macro_mae": v["macro_mae"],
        "test_macro_mae": t["macro_mae"],
        "test_nocyl_macro_mae": t.get("nocyl_macro_mae"),
        "calib_k5_objmacro": calib_k5(run),
        "test_cosine_median": t.get("cosine_median"),
        "test_macro_aligned_mae": t.get("macro_aligned_mae"),
        "test_macro_peak_hit": t.get("macro_peak_hit"),
        "test_amp_logratio_median": t.get("amp_logratio_median"),
        "test_ssim": t.get("micro_ssim"),
        **{g: t.get(f"group_{g}_mae") for g in GROUPS},
    }
    r["test_objmacro_mae"] = float(np.mean([r[g] for g in GROUPS]))
    if source_split:
        r |= per_source_val(bundle)
    return r


rows = [row(b, "indenter+grasp", True) for b in MERGED]
rows += [row(b, "indenter only", False) for b in PHASE0]
(OUT / "baseline_runs.json").write_text(json.dumps(rows, indent=1))

cols = [
    ("run", "run"), ("train_data", "train data"), ("best_epoch", "best ep"),
    ("val_macro_mae", "val MAE"), ("val_indenter_micro_mae", "val indenter"),
    ("val_grasp_micro_mae", "val grasp"), ("test_macro_mae", "test MAE"),
    ("test_objmacro_mae", "test 4-group mean"), ("calib_k5_objmacro", "calib k=5"),
    *[(g, g.replace("_TestXS_20", "").replace("_TestG_20", "").replace("_235", "")) for g in GROUPS],
    ("test_cosine_median", "cos"), ("test_macro_aligned_mae", "aligned"),
    ("test_macro_peak_hit", "peak hit"), ("test_amp_logratio_median", "amp log-ratio"),
    ("test_ssim", "SSIM"),
]


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if isinstance(x, float):
        return f"{x:.3f}" if abs(x) < 2 else f"{x:.1f}"
    return str(x)


lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
lines += ["| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |" for r in rows]
(OUT / "baseline_runs.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
