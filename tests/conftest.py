"""Shared synthetic-data fixtures for the taxelnet test suite.

None of this depends on real experimental data: every fixture builds small matched
real/simulation CSVs with the training (indenter) schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _helpers import N_NODES, write_grasp_pair, write_pair, write_synthetic_pair


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    """The 4-row ROWS pair: rows 0 and 2 share a key, sim values differ per row."""
    return write_pair(
        tmp_path, sim_value=lambda i: float(i) + np.arange(N_NODES) * 1e-3
    )


@pytest.fixture
def synthetic_pair(tmp_path: Path) -> tuple[Path, Path]:
    return write_synthetic_pair(tmp_path)


@pytest.fixture
def grasp_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A test-schema (grasp) pair in its own subdirectory."""
    return write_grasp_pair(tmp_path / "test")
