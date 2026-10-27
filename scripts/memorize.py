"""Memorisation check: fit ALL grasp training rows, no val, no weight decay, no selection.

Usage: python scripts/memorize.py --out DIR [--base-channels 16] [--epochs 1000] [--lr 1e-3]
Logs train MAE (eval mode, raw TC, overall and per pad) every 25 epochs to DIR/history.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from taxelnet.config import Configs
from taxelnet.dataset import GroupBatchSampler, TaxelDataset, load_paired
from taxelnet.evaluation import predict
from taxelnet.model import build_model
from taxelnet.preprocessing import Preprocessor
from taxelnet.training import train_one_epoch

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--data-dir", type=Path, default=Path("data_sensor/all"))
ap.add_argument("--base-channels", type=int, default=16)
ap.add_argument("--epochs", type=int, default=1000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(a.seed)
np.random.seed(a.seed)

c = Configs()
c.data.data_dir, c.data.real_csv = a.data_dir, "Dataset_GraspOnly_Real.csv"
c.data.sim_csv, c.data.cache_dir = "Dataset_GraspOnly_Simulation.csv", Path("data_sensor/.cache")
c.model.base_channels = a.base_channels
paired = load_paired(c.data)
rows = np.arange(len(paired.group))
pre = Preprocessor.fit(c.preprocessing, paired.x, paired.y, paired.sensor)
ds = TaxelDataset(paired, rows, pre.transform_x, pre.transform_y).to(c.train.device)
model = build_model(c.model, paired.taxel_shape).to(c.train.device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
sampler = GroupBatchSampler(ds.group.cpu().numpy(), 64, seed=a.seed)
sens = np.array([g["path"][-2:] for g in (paired.groups[i] for i in paired.group)])
hist, t0 = [], time.time()
for ep in range(1, a.epochs + 1):
    terms = train_one_epoch(model, ds, sampler, opt, 0.0, pre)
    sched.step()
    if ep % 25 == 0 or ep == a.epochs or ep == 1:
        pred = predict(model, ds, pre)
        err = np.abs(pred - paired.y).reshape(len(rows), -1).mean(1)
        h = {"epoch": ep, "loss": terms["l1"], "mae": float(err.mean()),
             "S1": float(err[sens == "S1"].mean()), "S2": float(err[sens == "S2"].mean())}
        hist.append(h)
        print(h, f"{time.time() - t0:.0f}s", flush=True)
        (a.out / "history.json").write_text(json.dumps({"args": vars(a) | {"out": str(a.out), "data_dir": str(a.data_dir)}, "history": hist}, indent=1))

# final predictions and the same errors split by "glitch" rows (any real taxel > 3000 TC)
pred = predict(model, ds, pre)
err = np.abs(pred - paired.y).reshape(len(rows), -1).mean(1)
zero = np.abs(paired.y).reshape(len(rows), -1).mean(1)
glitch = paired.y.reshape(len(rows), -1).max(1) > 3000
summary = {"n": len(rows), "n_glitch": int(glitch.sum())}
for pad in ("S1", "S2"):
    for gname, sel in (("all", np.ones_like(glitch)), ("clean", ~glitch), ("glitch", glitch)):
        m = (sens == pad) & sel
        summary[f"{pad}_{gname}"] = {"n": int(m.sum()), "mae": float(err[m].mean()) if m.any() else None,
                                    "zero_mae": float(zero[m].mean()) if m.any() else None}
summary["clean_overall"] = {"mae": float(err[~glitch].mean()), "zero_mae": float(zero[~glitch].mean())}
(a.out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
