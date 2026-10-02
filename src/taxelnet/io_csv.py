"""Schema-aware CSV -> GridTable parsing for mesh and taxel grid files.

Not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GridTable:
    keys: list[tuple[str, str]]
    forces: np.ndarray
    values: np.ndarray
    grid_shape: tuple[int, int]


def read_grid_csv(path: Path, grid_shape: tuple[int, int] | None = None) -> GridTable:
    raise NotImplementedError
