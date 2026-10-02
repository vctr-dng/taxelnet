"""Shared constants/value-functions for synthetic test data.

Plain helper module (not a conftest) so both conftest.py and individual test
modules can import it directly without relying on package-relative import
mode. Keep this free of pytest fixtures.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Mesh grid is deliberately much finer than the taxel grid, matching the
# real setup (dense simulation mesh -> coarse physical taxel array).
MESH_GRID_SHAPE = (4, 5)  # (H, W)
TAXEL_GRID_SHAPE = (2, 3)  # (H, W)

SAMPLE_KEYS = [
    ("obj_a", "1.0"),
    ("obj_a", "2.0"),
    ("obj_b", "1.0"),
    ("obj_b", "2.0"),
]


def mesh_value(sample_idx: int, row: int, col: int) -> float:
    return float(sample_idx * 100 + row * 10 + col)


def taxel_value(sample_idx: int, row: int, col: int) -> float:
    return float(sample_idx * 1000 + row * 10 + col)


def write_grid_csv(
    path: Path,
    keys: list[tuple[str, str]],
    grid_shape: tuple[int, int],
    value_fn,
) -> None:
    height, width = grid_shape
    columns = [f"r{r}_c{c}" for r in range(height) for c in range(width)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_name", "force", *columns])
        for sample_idx, (object_name, force) in enumerate(keys):
            row_values = [
                value_fn(sample_idx, r, c) for r in range(height) for c in range(width)
            ]
            writer.writerow([object_name, force, *row_values])


def expected_grid_array(
    keys: list[tuple[str, str]], grid_shape: tuple[int, int], value_fn
) -> np.ndarray:
    height, width = grid_shape
    out = np.zeros((len(keys), height, width), dtype=np.float32)
    for sample_idx in range(len(keys)):
        for r in range(height):
            for c in range(width):
                out[sample_idx, r, c] = value_fn(sample_idx, r, c)
    return out
