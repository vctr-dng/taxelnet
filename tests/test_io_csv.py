"""Tests for taxelnet.io_csv: matched real/simulation CSV -> GridTable readers."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import N_NODES, N_TAXELS, ROWS, write_pair

from taxelnet.io_csv import (
    SIM_GRID_SHAPE,
    TAXEL_GRID_SHAPE,
    TEST_KEY,
    read_real_table,
    read_sim_table,
    sim_node_columns,
    taxel_columns,
)


def test_read_real_table_reshapes_row_major(tmp_path):
    real_csv, _ = write_pair(
        tmp_path, real_value=lambda i: np.arange(N_TAXELS) + 100 * i
    )
    table = read_real_table(real_csv)

    assert table.grid_shape == TAXEL_GRID_SHAPE
    assert table.values.shape == (len(ROWS), *TAXEL_GRID_SHAPE)
    assert table.values.dtype == np.float32
    assert table.values[1, 0, 0] == 100  # tax1 -> (0, 0)
    assert table.values[1, 0, 3] == 103  # tax4 ends the first row
    assert table.values[1, 1, 0] == 104  # tax5 -> (1, 0)
    assert table.keys[1] == (5.1234, "TriangleM/Lateral2")
    assert table.dataset == ["02", "02", "02", "03"]  # zero-padded string, not int
    np.testing.assert_array_equal(table.source_rows, np.arange(len(ROWS)))


def test_read_sim_table_reshapes_row_major(tmp_path):
    _, sim_csv = write_pair(tmp_path, sim_value=lambda i: np.arange(N_NODES) + i)
    table = read_sim_table(sim_csv)

    assert table.grid_shape == SIM_GRID_SHAPE
    assert table.values.shape == (len(ROWS), *SIM_GRID_SHAPE)
    assert table.values[0, 0, 0] == 0  # n195 -> (0, 0)
    assert table.values[0, 0, 48] == 48  # n3 ends the first row
    assert table.values[0, 1, 0] == 49
    assert table.values[3, -1, -1] == N_NODES - 1 + 3  # n14899


def test_repeated_rows_are_kept(tmp_path):
    table = read_real_table(write_pair(tmp_path)[0])
    assert table.keys[0] == table.keys[2]
    assert len(table.keys) == len(ROWS)


def test_read_sim_table_rejects_wrong_node_order(tmp_path):
    cols = sim_node_columns()
    cols[0], cols[1] = cols[1], cols[0]
    _, sim_csv = write_pair(tmp_path, node_cols=cols)
    with pytest.raises(ValueError, match="row-major"):
        read_sim_table(sim_csv)


def test_read_sim_table_rejects_rotated_export(tmp_path):
    _, sim_csv = write_pair(tmp_path, node_cols=sim_node_columns()[::-1])
    with pytest.raises(ValueError, match="rotated"):
        read_sim_table(sim_csv)


def test_read_sim_table_rejects_missing_node(tmp_path):
    _, sim_csv = write_pair(tmp_path, node_cols=sim_node_columns()[:-1])
    with pytest.raises(ValueError, match="3772 columns"):
        read_sim_table(sim_csv)


def test_read_real_table_rejects_missing_taxel(tmp_path):
    real_csv, _ = write_pair(tmp_path, tax_cols=taxel_columns()[:-1])
    with pytest.raises(ValueError, match="1 missing"):
        read_real_table(real_csv)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_read_real_table_rejects_non_finite(tmp_path, bad):
    real_csv, _ = write_pair(tmp_path, real_value=lambda i: bad if i == 3 else 1.0)
    with pytest.raises(ValueError, match=r"non-finite values, e.g. data rows \[3\]"):
        read_real_table(real_csv)


def test_read_sim_table_rejects_non_finite(tmp_path):
    _, sim_csv = write_pair(tmp_path, sim_value=lambda i: np.nan if i == 1 else 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        read_sim_table(sim_csv)


def test_readers_reject_forces_beyond_4_decimals(tmp_path):
    rows = [*ROWS[:3], ("3.50001", "AForm/Horizontal", "03")]
    real_csv, sim_csv = write_pair(tmp_path, sim_rows=rows, real_rows=rows)
    with pytest.raises(ValueError, match="4 decimals"):
        read_real_table(real_csv)
    with pytest.raises(ValueError, match="4 decimals"):
        read_sim_table(sim_csv)


def test_read_sim_table_cache_hit(tmp_path):
    _, sim_csv = write_pair(
        tmp_path, sim_value=lambda i: np.arange(N_NODES) * 0.001 + i
    )
    cache = tmp_path / "cache"
    first = read_sim_table(sim_csv, cache_dir=cache)
    (cached,) = cache.glob("sim_*.npy")
    second = read_sim_table(sim_csv, cache_dir=cache)

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(
        np.load(cached).reshape(first.values.shape), first.values
    )


def test_readers_accept_test_schema(grasp_pair):
    """Grasp files: `force, Name` key, no `dataset`, extra `std*` columns ignored."""
    real_csv, sim_csv = grasp_pair
    real = read_real_table(real_csv, TEST_KEY)
    sim = read_sim_table(sim_csv, TEST_KEY)

    assert real.values.shape == (20, *TAXEL_GRID_SHAPE)
    assert sim.values.shape == (20, *SIM_GRID_SHAPE)
    assert real.keys == sim.keys
    assert real.keys[0] == (235.0, "Sphere_TestG_235_0_S1")
    assert set(real.dataset) == set(sim.dataset) == {""}
