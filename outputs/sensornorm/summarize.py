"""Aggregate the sensornorm arms (A-D x seeds) into outputs/sensornorm/results.{json,md}.

Per run: metrics.json, test MAE per sensor (predictions_test.parquet), val MAE per sensor
(recomputed from model.pt), calibrated k=5 object-macro MAE (calib/*/calibration.json from
scripts/calibrate_eval.py, if present). Then mean +- std over seeds per arm.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl

from taxelnet.checkpoint import load_model
from taxelnet.config import DataConfig
from taxelnet.dataset import TaxelDataset, load_paired
from taxelnet.evaluation import predict
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import make_split

HERE = Path(__file__).parent
RUNS = HERE / "runs"
CALIB = {
    p.parent.name.split("_", 1)[1]: p for p in (HERE / "calib").glob("*/calibration.json")
}
GROUPS = ["Cuboid_TestXS_20", "Sphere_TestG_20", "Cylinder_TestM_235", "Cylinder_TestButtons_235"]
ARMS = {
    "A": "A control: global_scale",
    "B": "B per-sensor global_scale",
    "C": "C per-sensor per_taxel_zscore",
    "D": "D = B, 16 glitch rows (>10k TC) dropped",
}


def sensor_of(path: str) -> str:
    return re.search(r"_(S\d)$", path).group(1)


def val_per_sensor(bundle: Path) -> dict[str, float]:
    d = json.loads((bundle / "config.json").read_text())["data"]
    data = DataConfig(
        data_dir=Path(d["data_dir"]), real_csv=d["real_csv"], sim_csv=d["sim_csv"],
        key_cols=tuple(d["key_cols"]), val_ratio=d["val_ratio"], split_seed=d["split_seed"],
        split_strategy=d["split_strategy"], cache_dir=Path(d["cache_dir"]),
    )  # fmt: skip
    paired = load_paired(data)
    rows = make_split(paired, data).rows["val"]
    pre = Preprocessor.load(bundle / "preprocessing.json")
    ds = TaxelDataset(paired, rows, pre.transform_x, pre.transform_y)
    pred = predict(load_model(bundle / "model.pt"), ds, pre)
    err = np.abs(pred - paired.y[rows]).reshape(len(rows), -1).mean(1)
    sens = paired.sensor[rows]
    return {f"val_{s}_mae": float(err[sens == s].mean()) for s in ("S1", "S2")}


def test_per_sensor(bundle: Path) -> dict[str, float]:
    df = pl.read_parquet(bundle / "predictions_test.parquet")
    err = np.abs(np.stack(df["residual"].to_list())).mean(1)
    zero = np.abs(np.stack(df["target"].to_list())).mean(1)
    sens = np.array([sensor_of(p) for p in df["path"]])
    out = {}
    for s in ("S1", "S2"):
        out[f"test_{s}_mae"] = float(err[sens == s].mean())
        out[f"test_{s}_zero_mae"] = float(zero[sens == s].mean())
    return out


def collect(bundle: Path) -> dict:
    m = json.loads((bundle / "metrics.json").read_text())
    t, v = m["test"], m["val"]
    run = bundle.name.split("_", 1)[1]
    r = {
        "run": run, "arm": run.split("-")[1], "best_epoch": m["best_epoch"],
        "val_macro_mae": v["macro_mae"], "test_macro_mae": t["macro_mae"],
        "test_zero_mae": t["reference"]["trivial_zero"]["macro_mae"],
        "test_cosine": t["cosine_median"], "test_aligned_mae": t["macro_aligned_mae"],
        "test_peak_hit": t["macro_peak_hit"], "test_ssim": t["micro_ssim"],
        "test_amp_logratio": t["amp_logratio_median"],
        **{g: t[f"group_{g}_mae"] for g in GROUPS},
        **{f"zero_{g}": t["reference"]["trivial_zero"][f"group_{g}_mae"] for g in GROUPS},
        **val_per_sensor(bundle), **test_per_sensor(bundle),
    }  # fmt: skip
    r["test_objmacro_mae"] = float(np.mean([r[g] for g in GROUPS]))
    c = CALIB.get(run)
    if c is not None:
        r["calib_k5"] = json.loads(c.read_text())["test_object_macro_mae"]["global_gain"]["5"]
    return r


COLS = [
    ("best_epoch", "best ep"), ("val_macro_mae", "val MAE"), ("val_S1_mae", "val S1"),
    ("val_S2_mae", "val S2"), ("test_macro_mae", "test MAE"),
    ("test_objmacro_mae", "4-grp mean"), ("calib_k5", "calib k=5"),
    ("test_S1_mae", "test S1"), ("test_S2_mae", "test S2"),
    ("Cuboid_TestXS_20", "Cuboid"), ("Sphere_TestG_20", "Sphere"),
    ("Cylinder_TestM_235", "Cyl M"), ("Cylinder_TestButtons_235", "Cyl Buttons"),
    ("test_cosine", "cos"), ("test_aligned_mae", "aligned"), ("test_peak_hit", "peak hit"),
    ("test_ssim", "SSIM"), ("test_amp_logratio", "amp logratio"),
]


def ms(vals: list[float]) -> str:
    a = np.array(vals, dtype=float)
    if abs(a.mean()) < 2:
        return f"{a.mean():.3f} ± {a.std(ddof=0):.3f}"
    return f"{a.mean():.1f} ± {a.std(ddof=0):.1f}"


def main() -> None:
    rows = [collect(b) for b in sorted(RUNS.glob("*_sn-*")) if (b / "metrics.json").exists()]
    (HERE / "results.json").write_text(json.dumps(rows, indent=1))
    lines = ["| arm | n | " + " | ".join(c[1] for c in COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for arm, label in ARMS.items():
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        cells = [ms([r[k] for r in rs]) if all(k in r for r in rs) else "–" for k, _ in COLS]
        lines.append(f"| {label} | {len(rs)} | " + " | ".join(cells) + " |")
    zero = rows[0]
    lines.append(
        "| zero predictor | – | " + " | ".join(
            {
                "test_macro_mae": f"{zero['test_zero_mae']:.1f}",
                "test_S1_mae": f"{zero['test_S1_zero_mae']:.1f}",
                "test_S2_mae": f"{zero['test_S2_zero_mae']:.1f}",
                **{g: f"{zero['zero_' + g]:.1f}" for g in GROUPS},
            }.get(k, "–") for k, _ in COLS
        ) + " |"
    )  # fmt: skip
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
