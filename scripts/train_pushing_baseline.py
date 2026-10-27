"""Pushing_task CNN (Training_CNN_06_01_v2) trained once on a fixed split, tested with and
without train/test leakage.

Data alignment, network and hyperparameters follow Pushing_task. The split comes from
scripts/make_pushing_split.py (train / val / test_clean / test_leaky). One model is
trained per seed on the same train and val rows; it is evaluated on val, test_clean
(no key shared with train) and test_leaky (keys shared with train). The best epoch is
selected on val loss only.

Usage: uv run python scripts/train_pushing_baseline.py [--seeds 0 1 2] [--epochs N]
"""

import argparse
import copy
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import nn

from taxelnet.evaluation import _micro_macro, global_ssim
from taxelnet.plots import select_plot_samples

ROWS, COLS, N_OUTPUTS, FORCE_DECIMALS = 18, 12, 28, 2
TAXEL_SHAPE = (7, 4)  # data_0..data_27 row-major, as tax1..tax28 in taxelnet
SSIM_DATA_RANGE = 3600.0  # TC, same as EvalConfig.ssim_data_range
PLOT_SAMPLES = 6
CONFIG = {
    "Conv2D_Init": 64,
    "Conv2D_0": 320,
    "Conv2D_1": 512,
    "Conv2D_2": 512,
    "Dense_0": 480,
    "Dense_1": 352,
    "learning_rate": 1e-4,
    "DropoutRate_0": 0.1,
    "batch_size": 120,
}


def load_and_align(raw_dir: Path):
    """One sample per real row: X = first ROWS sim rows of its (force, path), y = 28 values."""
    real = pl.read_csv(raw_dir / "Dataset_training_real.csv")
    sim = pl.read_csv(raw_dir / "Dataset_training_simulation.csv")
    pixel_cols = [str(i) for i in range(COLS)]
    real_cols = [f"data_{i}" for i in range(N_OUTPUTS)]

    def keyed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("force").round(FORCE_DECIMALS).alias("force_r"),
            pl.col("path").cast(pl.Utf8).str.strip_chars().alias("path_s"),
        )

    real, sim = keyed(real), keyed(sim)
    sim = sim.sort(["force_r", "path_s", "row_index"])
    grids: dict[tuple[float, str], np.ndarray] = {}
    for (f, p), g in sim.group_by(["force_r", "path_s"], maintain_order=True):
        if g.height >= ROWS:
            grids[(float(f), str(p))] = g.head(ROWS)[pixel_cols].to_numpy().astype(np.float32)

    keys = list(zip(real["force_r"].to_list(), real["path_s"].to_list()))
    keep = [i for i, k in enumerate(keys) if k in grids]
    if len(keep) < len(keys):
        print(f"[WARN] {len(keys) - len(keep)} real rows had no sim match")
    keys = [keys[i] for i in keep]
    X = np.stack([grids[k] for k in keys])[:, None]  # (N, 1, ROWS, COLS)
    y = real[real_cols].to_numpy().astype(np.float32)[keep]
    return X, y, keys


class PushingCNN(nn.Module):
    """Keras model port: 2x2 'same' convs (extra pad at the end), 'same' max pools."""

    def __init__(self, c: dict):
        super().__init__()
        self.c1 = nn.Conv2d(1, c["Conv2D_Init"], 2)
        self.c2 = nn.Conv2d(c["Conv2D_Init"], c["Conv2D_0"], 2)
        self.c3 = nn.Conv2d(c["Conv2D_0"], c["Conv2D_1"], 2)
        self.c4 = nn.Conv2d(c["Conv2D_1"], c["Conv2D_2"], 2)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.d0 = nn.Linear(c["Conv2D_2"], c["Dense_0"])
        self.bn = nn.BatchNorm1d(c["Dense_0"], eps=1e-3, momentum=0.01)
        self.drop = nn.Dropout(c["DropoutRate_0"])
        self.d1 = nn.Linear(c["Dense_0"], c["Dense_1"])
        self.out = nn.Linear(c["Dense_1"], N_OUTPUTS)

    @staticmethod
    def _same(conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (0, 1, 0, 1)))

    def forward(self, x):
        x = self.pool(self._same(self.c1, x))
        x = self._same(self.c2, x).relu()
        x = self.pool(self._same(self.c3, x).relu())
        x = self.pool(self._same(self.c4, x).relu())
        x = x.mean(dim=(2, 3))
        x = self.drop(self.bn(self.d0(x).relu()))
        return self.out(self.d1(x).relu())


def _cell_text(ax, grid: np.ndarray, fmt: str = "{:.0f}") -> None:
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax.text(c, r, fmt.format(grid[r, c]), ha="center", va="center", fontsize=6)


def plot_samples(x, target, pred, titles, epoch: int) -> Figure:
    """One row per test reading: sim input, target taxels, predicted taxels, residual."""
    n = len(target)
    fig = Figure(figsize=(11, 3.4 * n), layout="constrained")
    axes = fig.subplots(n, 4, squeeze=False)
    for i in range(n):
        t, p = target[i].reshape(TAXEL_SHAPE), pred[i].reshape(TAXEL_SHAPE)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        res = p - t
        lim = max(float(np.abs(res).max()), 1e-6)
        ax_x, ax_t, ax_p, ax_r = axes[i]
        fig.colorbar(ax_x.imshow(x[i, 0], cmap="viridis"), ax=ax_x)
        ax_x.set_title(titles[i], fontsize=8)
        for ax, g, name in ((ax_t, t, "target"), (ax_p, p, "prediction")):
            im = ax.imshow(g, cmap="magma", vmin=lo, vmax=hi)
            _cell_text(ax, g)
            ax.set_title(name)
        fig.colorbar(im, ax=[ax_t, ax_p], label="TC")
        im = ax_r.imshow(res, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _cell_text(ax_r, res)
        fig.colorbar(im, ax=ax_r)
        ax_r.set_title(f"residual (MAE {np.abs(res).mean():.1f})")
    fig.suptitle(f"epoch {epoch}")
    return fig


PARTS = ("val", "test_clean", "test_leaky")
# clean sets restricted to (force, path) keys with >= 2 real readings, the only kind of key
# test_leaky can contain: removes the group-mix confound when comparing clean vs leaky
EVAL_PARTS = PARTS + ("val_multi", "test_clean_multi")


def predict(model, X) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i : i + 1024]) for i in range(0, len(X), 1024)])


def evaluate(pred: np.ndarray, target: np.ndarray, group: np.ndarray, prefix: str) -> dict:
    """Micro (per reading) and macro (per (force, path) group) loss, MAE and SSIM."""
    err = np.abs(pred - target)
    ssim = global_ssim(
        pred.reshape(-1, *TAXEL_SHAPE), target.reshape(-1, *TAXEL_SHAPE), SSIM_DATA_RANGE
    )
    m = {}
    for name, per_reading in (
        ("loss", err.sum(1)),  # L1-sum, the training loss
        ("mae", err.mean(1)),
        ("ssim", ssim),
    ):
        for k, v in _micro_macro(name, per_reading, group).items():
            m[f"{prefix}_{k}"] = v
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_json", type=Path, default=Path("data/splits/pushing_split.json"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])  # model init / batch order
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--test_every", type=int, default=10)
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--experiment", default="pushing-leakage-v2")
    args = ap.parse_args()

    X, y, keys = load_and_align(args.raw_dir)
    split = json.loads(args.split_json.read_text())
    assert split["n_rows"] == len(keys), "split json does not match the real CSV"
    rows = {p: np.array(r) for p, r in split["rows"].items()}
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    group = np.array([uniq[k] for k in keys])
    train_keys = {keys[i] for i in rows["train"]}
    n_per_group = np.bincount(group)
    for p in ("val", "test_clean"):
        rows[f"{p}_multi"] = rows[p][n_per_group[group[rows[p]]] >= 2]
    for p in EVAL_PARTS:
        seen = sum(keys[i] in train_keys for i in rows[p])
        print(f"{p}: {len(rows[p])} rows, {seen} with key in train")

    x_max = X[rows["train"]].max()  # normalise by the training max, as Pushing_task
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X / x_max).to(dev)
    yt = torch.from_numpy(y).to(dev)
    idx = {p: torch.from_numpy(r).to(dev) for p, r in rows.items()}
    y_np = {p: y[rows[p]] for p in EVAL_PARTS}
    g_np = {p: group[rows[p]] for p in EVAL_PARTS}
    bs = CONFIG["batch_size"]

    def eval_part(model, p: str):
        pred = predict(model, Xt[idx[p]]).cpu().numpy()
        return evaluate(pred, y_np[p], g_np[p], p), pred

    # same readings in every figure and every run
    plot_pos = {p: select_plot_samples(y_np[p], PLOT_SAMPLES) for p in PARTS}

    def titles(p: str):
        return [
            f"{p}: {keys[rows[p][j]][1]} {keys[rows[p][j]][0]:g} N"
            + (" [key in train]" if keys[rows[p][j]] in train_keys else " [key unseen]")
            for j in plot_pos[p]
        ]

    def log_plots(model_preds: dict, folder: str, epoch: int):
        for p in ("test_clean", "test_leaky"):
            pos = plot_pos[p]
            fig = plot_samples(
                Xt[idx[p][torch.from_numpy(pos).to(dev)]].cpu().numpy(),
                y_np[p][pos], model_preds[p][pos], titles(p), epoch,
            )
            mlflow.log_figure(fig, f"plots/{folder}/{p}_samples.png")

    Path("outputs").mkdir(exist_ok=True)
    mlflow.set_experiment(args.experiment)
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = PushingCNN(CONFIG).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        best = {"val_loss": float("inf"), "epoch": 0, "state": None}

        with mlflow.start_run(run_name=f"split{split['seed']}-seed{seed}"):
            mlflow.log_params(
                {
                    **CONFIG,
                    "seed": seed,
                    "split_seed": split["seed"],
                    "split_sha_rows": split["real_csv_sha256"][:12],
                    "epochs": args.epochs,
                    **{f"n_{p}_rows": len(rows[p]) for p in ("train", *PARTS)},
                }
            )
            mlflow.log_artifact(str(args.split_json))
            mlflow.log_params({f"n_{p}_rows": len(rows[p]) for p in EVAL_PARTS if p.endswith("_multi")})

            for epoch in range(1, args.epochs + 1):
                model.train()
                order = idx["train"][torch.randperm(len(idx["train"]), device=dev)]
                total = 0.0
                for i in range(0, len(order), bs):
                    b = order[i : i + bs]
                    loss = (model(Xt[b]) - yt[b]).abs().sum(dim=1).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total += loss.item() * len(b)
                train_loss = total / len(order)

                val_pred = predict(model, Xt[idx["val"]])
                val_loss = (val_pred - yt[idx["val"]]).abs().sum(dim=1).mean().item()
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
                if val_loss < best["val_loss"]:
                    best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

                if epoch % args.test_every == 0 or epoch == args.epochs:
                    metrics, preds = {}, {}
                    for p in EVAL_PARTS:
                        m, preds[p] = eval_part(model, p)
                        metrics |= m
                    mlflow.log_metrics(metrics, step=epoch)
                    log_plots(preds, f"epoch_{epoch:04d}", epoch)
                    print(
                        f"split {split['seed']} seed {seed} epoch {epoch} train {train_loss:.1f} val {val_loss:.1f} "
                        f"clean mae {metrics['test_clean_micro_mae']:.2f} ssim {metrics['test_clean_micro_ssim']:.4f} | "
                        f"leaky mae {metrics['test_leaky_micro_mae']:.2f} ssim {metrics['test_leaky_micro_ssim']:.4f}",
                        flush=True,
                    )

            # final: the best-val epoch (selection never uses a test set)
            model.load_state_dict(best["state"])
            final, preds = {}, {}
            for p in EVAL_PARTS:
                m, preds[p] = eval_part(model, p)
                final |= m
            mlflow.log_metric("best_epoch", best["epoch"])
            mlflow.log_metrics({f"final_{k}": v for k, v in final.items()})
            log_plots(preds, "best", best["epoch"])
            torch.save({"state": best["state"], "x_max": float(x_max), "epoch": best["epoch"]}, Path(f"outputs/pushing_split{split['seed']}_seed{seed}_best.pt"))
            print(f"split {split['seed']} seed {seed} best epoch {best['epoch']}:", json.dumps({k: round(v, 4) for k, v in final.items()}))
            print("MLflow run", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
