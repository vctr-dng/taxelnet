"""Tests for taxelnet.splits: weighted group allocation and the immutable test manifest."""

from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import write_synthetic_pair

from taxelnet.config import DataConfig
from taxelnet.dataset import load_paired
from taxelnet.splits import PARTITIONS, allocate_groups, make_split, split_manifest


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
        data_dir=tmp_path,
        real_csv=real_csv.name,
        sim_csv=sim_csv.name,
        cache_dir=None,
        test_manifest=tmp_path / "splits" / "test_manifest.json",
    )
    return load_paired(cfg), cfg


def test_make_split_partitions_groups(paired_cfg):
    paired, cfg = paired_cfg
    split = make_split(paired, cfg, "sha")

    all_rows = np.concatenate([split.rows[p] for p in PARTITIONS])
    assert sorted(all_rows.tolist()) == list(range(len(paired.group)))
    groups = {p: set(paired.group[split.rows[p]].tolist()) for p in PARTITIONS}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    n = len(paired.group)
    assert abs(len(split.rows["test"]) - 0.1 * n) <= 3
    assert abs(len(split.rows["val"]) - 0.1 * n) <= 3
    assert split.test_manifest_created
    # repeated groups are held out too, not all allocated to train
    for p in ("val", "test"):
        assert (np.bincount(paired.group[split.rows[p]]) >= 2).any()


def test_existing_test_manifest_is_reused(paired_cfg):
    paired, cfg = paired_cfg
    first = make_split(paired, cfg, "sha")
    before = cfg.test_manifest.read_text()

    cfg.split_seed = 99  # a different seed must not move the test set
    second = make_split(paired, cfg, "sha")

    assert not second.test_manifest_created
    assert cfg.test_manifest.read_text() == before
    np.testing.assert_array_equal(first.rows["test"], second.rows["test"])
    assert abs(len(second.rows["val"]) - 0.1 * len(paired.group)) <= 3


def test_test_manifest_from_other_data_is_rejected(paired_cfg):
    paired, cfg = paired_cfg
    make_split(paired, cfg, "sha")
    with pytest.raises(ValueError, match="different real CSV"):
        make_split(paired, cfg, "other-sha")


def test_test_manifest_with_unknown_key_is_rejected(paired_cfg):
    paired, cfg = paired_cfg
    make_split(paired, cfg, "sha")
    manifest = json.loads(cfg.test_manifest.read_text())
    manifest["keys"].append([123.0, "Nope/Center"])
    cfg.test_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="1 test keys are not in the data"):
        make_split(paired, cfg, "sha")


def test_split_manifest_records_every_group(paired_cfg):
    paired, cfg = paired_cfg
    split = make_split(paired, cfg, "sha")
    manifest = split_manifest(paired, split, cfg)

    assert len(manifest["groups"]) == len(paired.groups)
    assert sum(g["n_real"] for g in manifest["groups"]) == len(paired.group)
    assert sum(c["readings"] for c in manifest["counts"].values()) == len(paired.group)
    shares = manifest["train_path_weight_share"].values()
    assert sum(s["loss_weight_share"] for s in shares) == pytest.approx(1.0)
