"""Synthetic matched real/simulation CSV pairs for the taxelnet test suite.

Plain helper module (not a conftest) so both conftest.py and individual test modules can
import it directly. Keep this free of pytest fixtures.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

from taxelnet.io_csv import (
    SIM_GRID_SHAPE,
    TAXEL_GRID_SHAPE,
    sim_node_columns,
    taxel_columns,
)

N_NODES = SIM_GRID_SHAPE[0] * SIM_GRID_SHAPE[1]
N_TAXELS = TAXEL_GRID_SHAPE[0] * TAXEL_GRID_SHAPE[1]

# (force, Path, dataset); rows 0 and 2 are repeated presses of one key pair
ROWS = [
    ("10.0", "CircleG/Center", "02"),
    ("5.1234", "TriangleM/Lateral2", "02"),
    ("10.0", "CircleG/Center", "02"),
    ("3.5", "AForm/Horizontal", "03"),
]

Values = Callable[[int], float | np.ndarray]


def _row_values(fn: Values, i: int, n: int) -> list:
    v = fn(i)
    return np.broadcast_to(np.asarray(v, dtype=float), (n,)).tolist()


def write_pair(
    tmp_path: Path,
    sim_rows=ROWS,
    sim_value: Values = lambda i: float(i % 2),
    node_cols: list[str] | None = None,
    real_rows=None,
    real_value: Values = float,
    tax_cols: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write real.csv and sim.csv; values per row are a scalar or a flat row-major array."""
    real_csv, sim_csv = tmp_path / "real.csv", tmp_path / "sim.csv"
    tax = tax_cols or taxel_columns()
    nodes = node_cols or sim_node_columns()
    with real_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["force", "Path", "dataset", *tax])
        for i, row in enumerate(real_rows or ROWS):
            writer.writerow([*row, *_row_values(real_value, i, len(tax))])
    with sim_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["force", "Path", "dataset", *nodes])
        for i, row in enumerate(sim_rows):
            writer.writerow([*row, *_row_values(sim_value, i, len(nodes))])
    return real_csv, sim_csv


def synthetic_rows(n_keys: int = 60, seed: int = 0) -> list[tuple[str, str, str]]:
    """Keys over a few paths; every fifth key is pressed 3 times."""
    rng = np.random.default_rng(seed)
    paths = [
        "CircleG/Center",
        "TriangleM/Lateral2",
        "AForm/Horizontal",
        "DonutM/Corner4",
    ]
    rows = []
    for k in range(n_keys):
        path = paths[k % len(paths)]
        row = (f"{1 + 0.25 * k:.2f}", path, f"0{2 + k % len(paths)}")
        rows += [row] * (3 if k % 5 == 0 else 1)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def write_synthetic_pair(tmp_path: Path, n_keys: int = 60, seed: int = 0):
    """A learnable pair: taxels are a block mean of the deformation grid plus noise."""
    rows = synthetic_rows(n_keys, seed)
    rng = np.random.default_rng(seed)
    sims = rng.random((len(rows), *SIM_GRID_SHAPE)) * 2.0 - 0.1
    blocks = sims[:, :77, :48].reshape(len(rows), 7, 11, 4, 12).mean(axis=(2, 4))
    reals = 1000.0 * blocks + rng.normal(0, 10, blocks.shape)
    return write_pair(
        tmp_path,
        sim_rows=rows,
        sim_value=lambda i: sims[i].ravel().round(3),
        real_rows=rows,
        real_value=lambda i: reals[i].ravel(),
    )
