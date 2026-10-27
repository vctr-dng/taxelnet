"""Inference plots for a checkpoint: sample maps, prediction scatter, per-taxel error.

Figures are built with the object-oriented matplotlib API (no pyplot), so importing this
module never changes the global backend. Maps are drawn in pad coordinates: the vertical
axis is X (41.80 mm at the top, like tax1 / n195), the horizontal axis is Y.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from taxelnet.io_csv import X_MAX, X_MIN, Y_MAX, Y_MIN

# imshow extent (left, right, bottom, top) in mm: row 0 at X_MAX, column 0 at Y_MAX
PAD_EXTENT = (Y_MAX, Y_MIN, X_MIN, X_MAX)


def select_plot_samples(target: np.ndarray, n: int) -> np.ndarray:
    """Positions of `n` readings spread from weak to strong peak signal; deterministic."""
    if n <= 0 or len(target) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(np.abs(target).reshape(len(target), -1).max(1), kind="stable")
    picks = np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(np.int64)
    return order[picks]


def plot_group_label(name: str, force: float) -> str:
    """Plot group of a reading: object, force, pad if any.

    The object is the Path part before "/" (training files) or the first two Name
    tokens (grasp test file): `Cylinder_TestButtons_235_..._S2` @ 235 ->
    `Cylinder_TestButtons_235_S2`; `Cuboid_G/20_02_A_..._S1` @ 20 -> `Cuboid_G_20_S1`;
    `TriangleM/Lateral2` @ 40 -> `TriangleM_40`.
    """
    tokens = re.split(r"[_/]", name)
    obj = name.split("/")[0] if "/" in name else "_".join(tokens[:2])
    label = f"{obj}_{force:g}"
    if tokens[-1] in ("S1", "S2"):
        label += f"_{tokens[-1]}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)


def select_group_samples(
    target: np.ndarray, labels: np.ndarray, n: int
) -> dict[str, np.ndarray]:
    """{label: positions of `n` readings of that group, weak to strong peak signal}."""
    out = {}
    for label in sorted(set(labels.tolist())):
        pos = np.flatnonzero(labels == label)
        out[label] = pos[select_plot_samples(target[pos], n)]
    return out


def plot_ssim_by_group(ssim: np.ndarray, labels: np.ndarray) -> Figure:
    """Per-reading SSIM of every group: box (quartiles, 1.5 IQR whiskers) + points."""
    groups = sorted(set(labels.tolist()))
    fig = Figure(figsize=(7, 1.2 + 0.55 * len(groups)), layout="constrained")
    ax = fig.subplots()
    data = [ssim[labels == g] for g in groups]
    ax.boxplot(
        data,
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        medianprops={"color": "#1f3a5f", "lw": 2},
        boxprops={"color": "#6b7280"},
        whiskerprops={"color": "#6b7280"},
        capprops={"color": "#6b7280"},
    )
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, start=1):
        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(d, i + jitter, s=8, color="#3b82c4", alpha=0.45, lw=0)
        ax.text(
            1.02, i, f"median {np.median(d):.2f}  (n={len(d)})",
            va="center", fontsize=8, color="#374151",
            transform=ax.get_yaxis_transform(),
        )  # fmt: skip
    ax.set_yticks(range(1, len(groups) + 1), groups, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SSIM (one window per map)")
    ax.grid(axis="x", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig


def _annotate(ax, im, grid: np.ndarray, fmt: str = "{:.0f}") -> None:
    """Write each cell's value, in black or white depending on the cell colour."""
    rgba = im.cmap(im.norm(grid))
    luminance = rgba[..., :3] @ np.array([0.299, 0.587, 0.114])
    h, w = grid.shape
    dy, dx = (Y_MIN - Y_MAX) / w, (X_MAX - X_MIN) / h
    for r in range(h):
        for c in range(w):
            ax.text(
                Y_MAX + (c + 0.5) * dy,
                X_MAX - (r + 0.5) * dx,
                fmt.format(grid[r, c]),
                ha="center",
                va="center",
                fontsize=6,
                color="black" if luminance[r, c] > 0.5 else "white",
            )


def plot_samples(
    x_mm: np.ndarray, target: np.ndarray, pred: np.ndarray, titles: list[str]
) -> Figure:
    """One row per reading: input deformation, target, prediction, residual."""
    n = len(target)
    fig = Figure(figsize=(12, 3.6 * max(n, 1)), layout="constrained")
    axes = fig.subplots(max(n, 1), 4, squeeze=False)
    for i in range(n):
        ax_in, ax_t, ax_p, ax_r = axes[i]
        im = ax_in.imshow(x_mm[i], extent=PAD_EXTENT, cmap="viridis")
        fig.colorbar(im, ax=ax_in, label="mm")
        lo = min(target[i].min(), pred[i].min())
        hi = max(target[i].max(), pred[i].max())
        for ax, grid, name in (
            (ax_t, target[i], "target"),
            (ax_p, pred[i], "prediction"),
        ):
            im = ax.imshow(grid, extent=PAD_EXTENT, cmap="magma", vmin=lo, vmax=hi)
            _annotate(ax, im, grid)
            ax.set_title(name)
        fig.colorbar(im, ax=[ax_t, ax_p], label="TC")
        residual = pred[i] - target[i]
        lim = max(float(np.abs(residual).max()), 1e-6)
        im = ax_r.imshow(
            residual, extent=PAD_EXTENT, cmap="RdBu_r", vmin=-lim, vmax=lim
        )
        _annotate(ax_r, im, residual)
        fig.colorbar(im, ax=ax_r, label="TC")
        ax_r.set_title(f"residual (MAE {np.abs(residual).mean():.1f})")
        ax_in.set_title(titles[i].replace(" @ ", "\n@ "), fontsize=8)
        ax_in.set_ylabel("X (mm)")
        for ax in axes[i]:
            ax.set_xlabel("Y (mm)")
    return fig


def plot_scatter(pred: np.ndarray, target: np.ndarray, metrics: dict) -> Figure:
    """Predicted vs target for every taxel of every reading."""
    fig = Figure(figsize=(6, 5.5), layout="constrained")
    ax = fig.subplots()
    t, p = target.ravel(), pred.ravel()
    hb = ax.hexbin(t, p, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    fig.colorbar(hb, ax=ax, label="taxels (log count)")
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color="tab:red", lw=1, label="y = x")
    ax.set_xlabel("target (TC)")
    ax.set_ylabel("prediction (TC)")
    ax.set_title(
        f"MAE micro {metrics['micro_mae']:.1f} / macro {metrics['macro_mae']:.1f} TC, "
        f"RMSE {metrics['micro_rmse']:.1f}",
        fontsize=9,
    )
    ax.legend(loc="upper left")
    return fig


def plot_per_taxel_mae(
    per_taxel_mae: list[float], taxel_shape: tuple[int, int]
) -> Figure:
    fig = Figure(figsize=(3.6, 5.5), layout="constrained")
    ax = fig.subplots()
    grid = np.asarray(per_taxel_mae).reshape(taxel_shape)
    im = ax.imshow(grid, extent=PAD_EXTENT, cmap="viridis", vmin=0)
    _annotate(ax, im, grid)
    fig.colorbar(im, ax=ax, label="MAE (TC)")
    ax.set_title("per-taxel MAE (micro)", fontsize=9)
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("X (mm)")
    return fig


def save_inference_plots(
    out_dir: Path,
    x_mm: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    metrics: dict,
    sample_pos: np.ndarray,
    sample_titles: list[str],
    title: str,
    group_labels: np.ndarray | None = None,
    ssim: np.ndarray | None = None,
    titles: list[str] | None = None,
    per_group: int = 3,
    max_groups: int = 40,
) -> list[Path]:
    """Write samples.png, scatter.png and per_taxel_mae.png; `x_mm`, `target`, `pred`
    cover the whole evaluated partition, `sample_pos` indexes into them.

    With `group_labels` (one per reading): also samples/<group>.png (`per_group`
    readings of each group, titled from `titles`, one per reading) and, with the
    per-reading `ssim`, ssim_by_group.png. Skipped when there are more than
    `max_groups` groups (e.g. indenter shape x force)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Figure] = {}
    if group_labels is not None and len(set(group_labels)) <= max_groups:
        labels = np.asarray(group_labels).astype(str)
        for label, pos in select_group_samples(target, labels, per_group).items():
            figures[f"samples/{label}.png"] = plot_samples(
                x_mm[pos], target[pos], pred[pos],
                [titles[p] if titles else label for p in pos],
            )  # fmt: skip
        if ssim is not None:
            figures["ssim_by_group.png"] = plot_ssim_by_group(np.asarray(ssim), labels)
    figures |= {
        "samples.png": plot_samples(
            x_mm[sample_pos], target[sample_pos], pred[sample_pos], sample_titles
        ),
        "scatter.png": plot_scatter(pred, target, metrics),
        "per_taxel_mae.png": plot_per_taxel_mae(
            metrics["per_taxel_micro_mae"], target.shape[1:]
        ),
    }
    paths = []
    for name, fig in figures.items():
        fig.suptitle(title)
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        paths.append(path)
    return paths
