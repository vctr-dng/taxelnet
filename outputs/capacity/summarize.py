"""Aggregate the capacity arms (arm x seeds) into outputs/capacity/results.{json,md}.

Needs, per run bundle: metrics.json, splits_eval.json (scripts/capacity_eval.py) and
optionally calib/<run>/calibration.json (scripts/calibrate_eval.py --out-root calib).
Mean +- std (population) over seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs"
CALIB = {p.parent.name.split("_", 1)[1]: p for p in (HERE / "calib").glob("*/calibration.json")}
TG = ["Cuboid_TestXS_20", "Sphere_TestG_20", "Cylinder_TestM_235", "Cylinder_TestButtons_235"]
ARMS = {
    "cnn_v0": "cnn_v0 (control)", "w32": "cnn_v0 base 32", "w64": "cnn_v0 base 64",
    "feat": "cnn_feat CoordConv", "featbig": "cnn_feat CoordConv big", "fp": "footprint (lr 1e-2)",
    "fpcnn": "footprint_cnn", "fpcnncoord": "footprint_cnn + CoordConv residual",
}


def collect(b: Path) -> dict:
    m = json.loads((b / "metrics.json").read_text())
    se = json.loads((b / "splits_eval.json").read_text())
    t, v = m["test"], m["val"]
    run = b.name.split("_", 1)[1]
    r = {"run": run, "arm": run.split("-")[1], "best_epoch": m["best_epoch"]}
    for tag in ("best", "last"):
        for part in ("train", "val", "test"):
            s = se[tag][part]
            r[f"{tag}_{part}"] = s["mae"]
            r[f"{tag}_{part}_macro"] = s["macro"]
            r[f"{tag}_{part}_S1"], r[f"{tag}_{part}_S2"] = s["S1"], s["S2"]
            for o, e in s["objects"].items():
                r[f"{tag}_{part}_obj_{o}"] = e
    r.update(
        test_macro=t["macro_mae"], val_macro=v["macro_mae"],
        zero=t["reference"]["trivial_zero"]["macro_mae"],
        cos=t["cosine_median"], aligned=t["macro_aligned_mae"], peak=t["macro_peak_hit"],
        ssim=t["micro_ssim"], **{g: t[f"group_{g}_mae"] for g in TG},
    )  # fmt: skip
    r["test_objmean"] = float(np.mean([r[g] for g in TG]))
    c = CALIB.get(run)
    if c is not None:
        r["calib_k5"] = json.loads(c.read_text())["test_object_macro_mae"]["global_gain"]["5"]
    return r


def ms(vals) -> str:
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f} ± {a.std():.3f}" if abs(a.mean()) < 2 else f"{a.mean():.1f} ± {a.std():.1f}"


def table(rows, cols) -> list[str]:
    out = ["| arm | n | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for arm, label in ARMS.items():
        rs = [r for r in rows if r["arm"] == arm]
        if rs:
            cells = [ms([r[k] for r in rs]) if all(k in r for r in rs) else "–" for k, _ in cols]
            out.append(f"| {label} | {len(rs)} | " + " | ".join(cells) + " |")
    return out


def main() -> None:
    rows = [collect(b) for b in sorted(RUNS.glob("*_cap-*")) if (b / "splits_eval.json").exists() and (b / "metrics.json").exists()]
    (HERE / "results.json").write_text(json.dumps(rows, indent=1))
    main_cols = [
        ("best_epoch", "best ep"), ("best_train", "train"), ("best_val", "val"), ("best_test", "test"),
        ("test_macro", "test (group macro)"), ("calib_k5", "calib k=5"), ("test_objmean", "4-grp mean"),
        ("cos", "cos"), ("aligned", "aligned"), ("peak", "peak hit"), ("ssim", "SSIM"),
    ]
    pad_cols = [
        (f"best_{p}_{s}", f"{p} {s}") for p in ("train", "val", "test") for s in ("S1", "S2")
    ]
    obj_cols = [(g, g.split("_")[0] + " " + g.split("_")[1].replace("Test", "")) for g in TG]
    last_cols = [("best_train", "train best"), ("last_train", "train last"), ("best_val", "val best"),
                 ("last_val", "val last"), ("best_test", "test best"), ("last_test", "test last")]
    shapes = sorted({k.split("_obj_")[1] for r in rows for k in r if k.startswith("best_train_obj_")})
    tr_obj = [(f"best_train_obj_{s}", s) for s in shapes]
    va_obj = sorted({k.split("_obj_")[1] for r in rows for k in r if k.startswith("best_val_obj_")})
    parts = [
        "### Main (best-val checkpoint; MAE in TC, micro over readings unless noted)", table(rows, main_cols),
        "### Per pad", table(rows, pad_cols),
        "### Test per object group", table(rows, obj_cols),
        "### Best vs last epoch", table(rows, last_cols),
        "### Train MAE per object", table(rows, tr_obj),
        "### Val MAE per held-out object", table(rows, [(f"best_val_obj_{s}", s) for s in va_obj]),
    ]
    (HERE / "results.md").write_text("\n\n".join("\n".join(p) if isinstance(p, list) else p for p in parts) + "\n")
    print((HERE / "results.md").read_text())


if __name__ == "__main__":
    main()
