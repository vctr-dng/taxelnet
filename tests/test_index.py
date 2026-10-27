"""Tests for taxelnet.dataset.build_index on small matched real/simulation CSV pairs."""

from __future__ import annotations

import pytest
from _helpers import ROWS, write_pair

from taxelnet.dataset import build_index
from taxelnet.io_csv import sim_node_columns


def test_sim_node_columns_match_readme_corners():
    cols = sim_node_columns()
    assert len(cols) == 77 * 49
    assert (cols[0], cols[48], cols[-49], cols[-1]) == (
        "n195",
        "n3",
        "n15091",
        "n14899",
    )


def test_build_index_lists_every_sim_row_one_to_one(tmp_path):
    index, n_sim_differ = build_index(*write_pair(tmp_path, sim_value=lambda i: 1.0))

    assert n_sim_differ == 0
    by_key = {(e["force"], e["path"]): e for e in index}
    assert by_key[(10.0, "CircleG/Center")]["real_id"] == [0, 2]
    assert by_key[(10.0, "CircleG/Center")]["dataset"] == "02"
    assert all(e["sim_id"] == e["real_id"] for e in index)
    assert sorted(i for e in index for i in e["real_id"]) == list(range(len(ROWS)))


def test_build_index_counts_differing_sim_rows_without_skipping(tmp_path):
    # rows 0 and 2 share a key but get sim values 0.0 and 2.0
    index, n_sim_differ = build_index(*write_pair(tmp_path, sim_value=float))

    assert n_sim_differ == 1
    assert len(index) == 3
    assert any(e["sim_id"] == [0, 2] for e in index)


def test_build_index_rejects_row_misaligned_files(tmp_path):
    with pytest.raises(ValueError, match="different rows"):
        build_index(
            *write_pair(tmp_path, sim_rows=[ROWS[1], ROWS[0], ROWS[2], ROWS[3]])
        )


def test_build_index_rejects_unmatched_key(tmp_path):
    sim_rows = [*ROWS[:3], ("3.5", "AForm/Vertical", "03")]
    with pytest.raises(ValueError, match="not in both files"):
        build_index(*write_pair(tmp_path, sim_rows=sim_rows))


def test_build_index_rejects_forces_beyond_4_decimals(tmp_path):
    sim_rows = [*ROWS[:3], ("3.50001", "AForm/Horizontal", "03")]
    with pytest.raises(ValueError, match="4 decimals"):
        build_index(*write_pair(tmp_path, sim_rows=sim_rows))


def test_build_index_rejects_rotated_sim_export(tmp_path):
    rotated = sim_node_columns()[::-1]  # Extracted_simulation_data.csv order
    with pytest.raises(ValueError, match="n195"):
        build_index(*write_pair(tmp_path, node_cols=rotated))
