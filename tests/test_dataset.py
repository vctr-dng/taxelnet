"""Tests for taxelnet.dataset: strict row pairing, TaxelDataset, GroupBatchSampler."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _helpers import ROWS, write_pair

from taxelnet.config import DataConfig
from taxelnet.dataset import GroupBatchSampler, TaxelDataset, load_paired


def _cfg(real_csv, sim_csv) -> DataConfig:
    return DataConfig(
        data_dir=real_csv.parent,
        real_csv=real_csv.name,
        sim_csv=sim_csv.name,
        cache_dir=None,
    )


def test_repeated_key_is_one_group_with_own_sim_rows(pair):
    paired = load_paired(_cfg(*pair))

    assert paired.x.shape == (len(ROWS), 77, 49)
    assert paired.y.shape == (len(ROWS), 7, 4)
    assert paired.group[0] == paired.group[2]
    assert len(set(paired.group.tolist())) == 3
    # each reading keeps its own simulation row, never a collapsed canonical grid
    assert paired.x[0, 0, 0] == pytest.approx(0.0)
    assert paired.x[2, 0, 0] == pytest.approx(2.0)
    assert paired.y[2, 0, 0] == 2.0
    assert paired.n_sim_differ == 1
    assert sorted(paired.group_sizes.tolist()) == [1, 1, 2]


def test_load_paired_rejects_row_misalignment(tmp_path):
    sim_rows = [ROWS[1], ROWS[0], ROWS[2], ROWS[3]]
    with pytest.raises(ValueError, match="2 rows have different force, Path, dataset"):
        load_paired(_cfg(*write_pair(tmp_path, sim_rows=sim_rows)))


def test_load_paired_rejects_row_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="has 4 rows but"):
        load_paired(_cfg(*write_pair(tmp_path, sim_rows=ROWS[:3])))


def test_load_paired_rejects_dataset_mismatch(tmp_path):
    sim_rows = [*ROWS[:3], ("3.5", "AForm/Horizontal", "04")]
    with pytest.raises(ValueError, match="different"):
        load_paired(_cfg(*write_pair(tmp_path, sim_rows=sim_rows)))


def test_taxel_dataset_items_and_batches(pair):
    paired = load_paired(_cfg(*pair))
    ds = TaxelDataset(paired, np.array([0, 2, 3]), transform_y=lambda y: y / 2)

    x, y, g = ds[1]
    assert x.shape == (1, 77, 49) and x.dtype == torch.float32
    assert y.shape == (7, 4)
    assert y[0, 0].item() == 1.0  # row 2 has value 2.0, halved
    assert g.item() == paired.group[2]
    xb, yb, gb = ds.__getitems__([0, 2])
    assert xb.shape == (2, 1, 77, 49) and yb.shape == (2, 7, 4) and gb.shape == (2,)


def test_group_batch_sampler_keeps_groups_whole():
    group = np.array([5, 1, 5, 2, 2, 2, 7, 1, 9])
    sampler = GroupBatchSampler(group, groups_per_batch=2, seed=0)

    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    seen = [i for b in batches for i in b]
    assert sorted(seen) == list(range(len(group)))
    for g in np.unique(group):
        holding = [k for k, b in enumerate(batches) if any(group[i] == g for i in b)]
        assert len(holding) == 1
    assert all(len(np.unique(group[b])) <= 2 for b in batches)


def test_group_batch_sampler_reshuffles_each_epoch():
    group = np.arange(50)
    sampler = GroupBatchSampler(group, groups_per_batch=10, seed=0)
    assert list(sampler) != list(sampler)
