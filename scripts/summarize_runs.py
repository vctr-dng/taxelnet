"""Arm table from run bundles' metrics.json: val, then test MAE / SSIM three ways.

Test pads are not the training pads, so test predictions are reported:
- uncal: with the training pad's y_scale (no adaptation);
- testscale: with the test pad's own y_scale (std of k of its real readings, or all);
- padgain: with one gain per test pad fitted on k of its readings (calibrated).
Arm = run name without the trailing -s<seed> / -f<fold>. Mean ± std over the arm's runs.

Usage: python scripts/summarize_runs.py RUNS_DIR [RUNS_DIR ...] [--out FILE.md]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

COLUMNS = (
    ("best ep", ("best_epoch",)),
    ("val MAE", ("val", "macro_mae")),
    ("val SSIM", ("val", "micro_ssim")),
    ("uncal MAE", ("test", "macro_mae")),
    ("uncal SSIM", ("test", "micro_ssim")),
    ("test-pad y_scale k5", ("test", "testscale_k5_mae")),
    ("k20", ("test", "testscale_k20_mae")),
    ("all*", ("test", "testscale_all_mae")),
    ("SSIM k20", ("test", "testscale_k20_ssim")),
    ("calib gain k5", ("test", "padgain_k5_mae")),
    ("k20", ("test", "padgain_k20_mae")),
    ("all*", ("test", "padgain_oracle_mae")),
    ("SSIM k20", ("test", "padgain_k20_ssim")),
    ("4-obj calib k5", ("test", "padgain_k5_obj_mean_mae")),
)


def get(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms: dict[str, list[dict]] = {}
    for runs in args.runs:
        for m in sorted(runs.glob("*/metrics.json")):
            name = m.parent.name.split("_", 1)[1]
            arm = re.sub(r"-[sf]\d+$", "", name)
            arm += " (5-fold)" if re.search(r"-f\d+$", name) else ""
            arms.setdefault(arm, []).append(json.loads(m.read_text()))
    head = "| arm | n | " + " | ".join(c for c, _ in COLUMNS) + " |"
    lines = [head, "|---|---|" + "---|" * len(COLUMNS)]
    for arm, ms in arms.items():
        cells = []
        for _, path in COLUMNS:
            v = np.array([x for x in (get(m, path) for m in ms) if x is not None], float)
            fmt = ".3f" if "ssim" in path[-1] else ".1f"
            cells.append(f"{v.mean():{fmt}} ± {v.std():{fmt}}" if len(v) else "–")
        lines.append(f"| {arm} | {len(ms)} | " + " | ".join(cells) + " |")
    lines.append("\n\\* all: computed on every reading of the test pad and scored on them")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
