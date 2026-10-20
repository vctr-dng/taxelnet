"""Train / val / test MAE of run bundles (best.pt and last.pt), overall, per pad, per object.

Usage: python scripts/capacity_eval.py BUNDLE [BUNDLE ...]
Writes BUNDLE/splits_eval.json: {"best"|"last": {"train"|"val"|"test": {mae, S1, S2, objects}}}.
Train rows are evaluated in eval mode (the models have no dropout / batch norm). Errors are
raw-TC per-reading MAE; "mae" is the micro mean over readings, "macro" the mean over
(force, Path) groups, objects are the shape (train/val) or test object group (test).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_eval import load_bundle_model, load_configs  # noqa: E402

from taxelnet.checkpoint import load_checkpoint  # noqa: E402
from taxelnet.dataset import TaxelDataset, load_paired, load_test_paired  # noqa: E402
from taxelnet.evaluation import object_group, predict  # noqa: E402
from taxelnet.model import build_model  # noqa: E402
from taxelnet.config import ModelConfig  # noqa: E402
from taxelnet.preprocessing import Preprocessor  # noqa: E402
from taxelnet.splits import make_split, shape_of  # noqa: E402


def summarize(err, sens, objs, group) -> dict:
    out = {"mae": float(err.mean()), "n": int(len(err))}
    _, inv = np.unique(group, return_inverse=True)
    out["macro"] = float((np.bincount(inv, weights=err) / np.bincount(inv)).mean())
    for s in ("S1", "S2"):
        out[s] = float(err[sens == s].mean()) if (sens == s).any() else None
    out["objects"] = {o: float(err[objs == o].mean()) for o in sorted(set(objs))}
    return out


def main() -> None:
    for bundle in map(Path, sys.argv[1:]):
        cfgs = load_configs(bundle)
        paired = load_paired(cfgs.data)
        test = load_test_paired(cfgs.data)
        split = make_split(paired, cfgs.data)
        pre = Preprocessor.load(bundle / "preprocessing.json")
        parts = {
            "train": (paired, split.rows["train"]),
            "val": (paired, split.rows["val"]),
            "test": (test, np.arange(len(test.group))),
        }
        res = {}
        for tag in ("best", "last"):
            model = build_model(ModelConfig(**vars(cfgs.model)), paired.taxel_shape)
            load_checkpoint(bundle / f"{tag}.pt", model)
            model = model.to(cfgs.train.device).eval()
            res[tag] = {}
            for name, (src, rows) in parts.items():
                ds = TaxelDataset(src, rows, pre.transform_x, pre.transform_y).to(
                    cfgs.train.device
                )
                pred = predict(model, ds, pre)
                err = np.abs(pred - src.y[rows]).reshape(len(rows), -1).mean(1)
                paths = [src.groups[g]["path"] for g in src.group[rows]]
                sens = np.array([re.search(r"_(S\d)$", p).group(1) for p in paths])
                objs = np.array(
                    [object_group(p) if name == "test" else shape_of(p) for p in paths]
                )
                res[tag][name] = summarize(err, sens, objs, src.group[rows])
            del model
            torch.cuda.empty_cache()
        (bundle / "splits_eval.json").write_text(json.dumps(res, indent=1))
        print(bundle.name, {t: {p: round(v["mae"], 1) for p, v in d.items()} for t, d in res.items()})


if __name__ == "__main__":
    main()
