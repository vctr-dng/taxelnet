"""Shared synthetic-data fixtures for the taxelnet test suite.

None of this depends on real experimental data: every fixture builds small,
deterministic CSVs (or in-memory samples) matching the schema assumptions
documented in the project plan, purely to exercise the data-loading, join,
model, and training-loop code paths.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from _helpers import (
    MESH_GRID_SHAPE,
    SAMPLE_KEYS,
    TAXEL_GRID_SHAPE,
    mesh_value,
    taxel_value,
    write_grid_csv,
)


@pytest.fixture
def mesh_grid_shape() -> tuple[int, int]:
    return MESH_GRID_SHAPE


@pytest.fixture
def taxel_grid_shape() -> tuple[int, int]:
    return TAXEL_GRID_SHAPE


@pytest.fixture
def sample_keys() -> list[tuple[str, str]]:
    return list(SAMPLE_KEYS)


@pytest.fixture
def tiny_mesh_csv(tmp_path: Path) -> Path:
    path = tmp_path / "mesh.csv"
    write_grid_csv(path, SAMPLE_KEYS, MESH_GRID_SHAPE, mesh_value)
    return path


@pytest.fixture
def tiny_taxel_csv(tmp_path: Path) -> Path:
    path = tmp_path / "taxel.csv"
    write_grid_csv(path, SAMPLE_KEYS, TAXEL_GRID_SHAPE, taxel_value)
    return path


@pytest.fixture
def mismatched_taxel_csv(tmp_path: Path) -> Path:
    """Taxel file missing one mesh key and containing one key mesh doesn't have."""
    keys = [
        ("obj_a", "1.0"),
        ("obj_a", "2.0"),
        ("obj_b", "1.0"),
        # ("obj_b", "2.0") deliberately dropped
        ("obj_c", "1.0"),  # not present in the mesh file at all
    ]
    path = tmp_path / "taxel_mismatched.csv"
    write_grid_csv(path, keys, TAXEL_GRID_SHAPE, taxel_value)
    return path


@pytest.fixture
def duplicate_key_mesh_csv(tmp_path: Path) -> Path:
    keys = [("obj_a", "1.0"), ("obj_a", "1.0")]  # duplicate (object_name, force)
    path = tmp_path / "mesh_duplicate.csv"
    write_grid_csv(path, keys, MESH_GRID_SHAPE, mesh_value)
    return path


@pytest.fixture
def malformed_header_csv(tmp_path: Path) -> Path:
    """Grid columns have a gap (missing r2_c2), so shape can't be cleanly inferred."""
    height, width = MESH_GRID_SHAPE
    columns = [
        f"r{r}_c{c}"
        for r in range(height)
        for c in range(width)
        if not (r == 2 and c == 2)
    ]
    path = tmp_path / "mesh_malformed.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_name", "force", *columns])
        writer.writerow(["obj_a", "1.0", *[1.0] * len(columns)])
    return path


@pytest.fixture
def flat_column_csv(tmp_path: Path) -> Path:
    """Grid columns that don't encode row/col in their names at all."""
    height, width = TAXEL_GRID_SHAPE
    num_cells = height * width
    columns = [f"taxel_{i}" for i in range(num_cells)]
    path = tmp_path / "taxel_flat.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object_name", "force", *columns])
        for sample_idx, (object_name, force) in enumerate(SAMPLE_KEYS):
            values = [
                taxel_value(sample_idx, i // width, i % width) for i in range(num_cells)
            ]
            writer.writerow([object_name, force, *values])
    return path
