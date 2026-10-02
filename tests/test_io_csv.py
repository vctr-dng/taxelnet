"""Tests for taxelnet.io_csv: schema-aware CSV -> GridTable parsing."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import (
    MESH_GRID_SHAPE,
    SAMPLE_KEYS,
    TAXEL_GRID_SHAPE,
    expected_grid_array,
    mesh_value,
    taxel_value,
)

from taxelnet.io_csv import read_grid_csv


def test_read_grid_csv_infers_mesh_shape_and_values(tiny_mesh_csv):
    table = read_grid_csv(tiny_mesh_csv)

    assert table.grid_shape == MESH_GRID_SHAPE
    assert table.keys == SAMPLE_KEYS
    assert table.values.shape == (len(SAMPLE_KEYS), *MESH_GRID_SHAPE)
    np.testing.assert_allclose(
        table.values, expected_grid_array(SAMPLE_KEYS, MESH_GRID_SHAPE, mesh_value)
    )


def test_read_grid_csv_infers_taxel_shape_and_values(tiny_taxel_csv):
    table = read_grid_csv(tiny_taxel_csv)

    assert table.grid_shape == TAXEL_GRID_SHAPE
    assert table.values.shape == (len(SAMPLE_KEYS), *TAXEL_GRID_SHAPE)
    np.testing.assert_allclose(
        table.values, expected_grid_array(SAMPLE_KEYS, TAXEL_GRID_SHAPE, taxel_value)
    )


def test_read_grid_csv_parses_force_as_float(tiny_taxel_csv):
    table = read_grid_csv(tiny_taxel_csv)

    assert table.forces.shape == (len(SAMPLE_KEYS),)
    np.testing.assert_allclose(table.forces, [float(force) for _, force in SAMPLE_KEYS])


def test_read_grid_csv_explicit_grid_shape_overrides_inference(flat_column_csv):
    table = read_grid_csv(flat_column_csv, grid_shape=TAXEL_GRID_SHAPE)

    assert table.grid_shape == TAXEL_GRID_SHAPE
    assert table.values.shape == (len(SAMPLE_KEYS), *TAXEL_GRID_SHAPE)
    np.testing.assert_allclose(
        table.values, expected_grid_array(SAMPLE_KEYS, TAXEL_GRID_SHAPE, taxel_value)
    )


def test_read_grid_csv_without_grid_shape_and_no_row_col_pattern_raises(
    flat_column_csv,
):
    with pytest.raises(ValueError):
        read_grid_csv(flat_column_csv)


def test_read_grid_csv_malformed_header_raises(malformed_header_csv):
    with pytest.raises(ValueError):
        read_grid_csv(malformed_header_csv)


def test_read_grid_csv_duplicate_keys_raises(duplicate_key_mesh_csv):
    with pytest.raises(ValueError):
        read_grid_csv(duplicate_key_mesh_csv)
