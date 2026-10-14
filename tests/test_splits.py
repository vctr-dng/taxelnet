"""Tests for taxelnet.splits: weighted group allocation and the train/val split."""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import write_synthetic_pair

from taxelnet.config import DataConfig
from taxelnet.dataset import load_paired, load_paired_files
from taxelnet.splits import (
    PARTITIONS,
    allocate_groups,
    make_split,
    shared_keys,
    split_manifest,
)


def test_allocate_groups_hits_reading_targets():
    rng = np.random.default_rng(0)
    sizes = np.concatenate([np.ones(900, int), rng.integers(2, 30, 60)])
    part = allocate_groups(sizes, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=3)

    total = sizes.sum()
    for name, ratio in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        assert abs(sizes[part == name].sum() - ratio * total) <= 1


def test_allocate_groups_is_seeded():
    sizes = np.random.default_rng(0).integers(1, 5, 200)
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = allocate_groups(sizes, ratios, seed=1)
    assert (a == allocate_groups(sizes, ratios, seed=1)).all()
    assert (a != allocate_groups(sizes, ratios, seed=2)).any()


@pytest.fixture
def paired_cfg(tmp_path):
    real_csv, sim_csv = write_synthetic_pair(tmp_path, n_keys=80)
    cfg = DataConfig(
        data_dir=tmp_path, real_csv=real_csv.name, sim_csv=sim_csv.name, cache_dir=None
    )
    return load_paired(cfg), cfg


def test_make_split_partitions_groups(paired_cfg):
    paired, cfg = paired_cfg
    split = make_split(paired, cfg)

    assert PARTITIONS == ("train", "val")
    all_rows = np.concatenate([split.rows[p] for p in PARTITIONS])
    assert sorted(all_rows.tolist()) == list(range(len(paired.group)))
    groups = {p: set(paired.group[split.rows[p]].tolist()) for p in PARTITIONS}
    assert groups["train"].isdisjoint(groups["val"])
    assert abs(len(split.rows["val"]) - 0.1 * len(paired.group)) <= 3
    # repeated groups are held out too, not all allocated to train
    assert (np.bincount(paired.group[split.rows["val"]]) >= 2).any()


def test_make_split_is_seeded(paired_cfg):
    paired, cfg = paired_cfg
    first = make_split(paired, cfg)
    np.testing.assert_array_equal(
        first.rows["val"], make_split(paired, cfg).rows["val"]
    )
    cfg.split_seed = 99
    assert not np.array_equal(first.rows["val"], make_split(paired, cfg).rows["val"])


def test_shared_keys(paired_cfg, grasp_pair):
    paired, _ = paired_cfg
    test = load_paired_files(grasp_pair[0], grasp_pair[1], ["force", "Name"])
    assert shared_keys(paired, test) == []
    assert shared_keys(paired, paired) == sorted(
        (float(r["force"]), r["path"]) for r in paired.groups
    )


def test_split_manifest_records_every_group(paired_cfg, grasp_pair):
    paired, cfg = paired_cfg
    test = load_paired_files(grasp_pair[0], grasp_pair[1], ["force", "Name"])
    split = make_split(paired, cfg)
    checksums = {f.name: "sha" for f in test.files}
    manifest = split_manifest(paired, split, cfg, test, checksums)

    assert len(manifest["groups"]) == len(paired.groups)
    assert sum(g["n_real"] for g in manifest["groups"]) == len(paired.group)
    assert sum(c["readings"] for c in manifest["counts"].values()) == len(paired.group)
    assert manifest["test"]["readings"] == len(test.group)
    assert set(manifest["test"]["files"]) == {"grasp_real.csv", "grasp_sim.csv"}
    shares = manifest["train_path_weight_share"].values()
    assert sum(s["loss_weight_share"] for s in shares) == pytest.approx(1.0)
