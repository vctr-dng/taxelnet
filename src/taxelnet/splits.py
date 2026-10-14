"""Seeded, weighted group-level train/val allocation and split manifests.

A full raw (force, Path) group is indivisible: all its presses go to one partition. Group
weight is its number of real readings. The test set is not carved from the training data:
it is the external pair in `DataConfig.test_dir`, identical for every run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from taxelnet.config import DataConfig
from taxelnet.dataset import PairedData

PARTITIONS = ("train", "val")  # carved from the training pair
TEST = "test"  # the external test pair
MANIFEST_VERSION = 2


def allocate_groups(
    group_sizes: np.ndarray, ratios: dict[str, float], seed: int
) -> np.ndarray:
    """Partition name per group, with reading counts as close as possible to `ratios`.

    Groups are shuffled (seeded), stable-sorted by size, largest first, then each one goes
    to the partition with the largest remaining deficit relative to its target reading
    count. The relative deficit spreads large repeated groups over every partition in
    proportion, instead of filling the largest partition with them first.
    """
    group_sizes = np.asarray(group_sizes)
    names = list(ratios)
    total = group_sizes.sum()
    targets = np.array([ratios[n] for n in names], dtype=float)
    targets = targets / targets.sum() * total
    counts = np.zeros(len(names))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(group_sizes))
    order = perm[np.argsort(-group_sizes[perm], kind="stable")]

    out = np.empty(len(group_sizes), dtype=object)
    for g in order:
        p = int(np.argmax((targets - counts) / targets))
        out[g] = names[p]
        counts[p] += group_sizes[g]
    return out


def _key(record: dict) -> tuple[float, str]:
    return (float(record["force"]), record["path"])


@dataclass(frozen=True)
class Split:
    partition: np.ndarray  # partition name per group
    rows: dict[str, np.ndarray]  # sorted data rows per partition


def make_split(paired: PairedData, data_cfg: DataConfig) -> Split:
    """Seeded train/val allocation of the training groups (no test carve-out)."""
    keys = [_key(r) for r in paired.groups]
    partition = allocate_groups(
        paired.group_sizes,
        {"train": 1 - data_cfg.val_ratio, "val": data_cfg.val_ratio},
        data_cfg.split_seed,
    )
    key_sets = {p: {k for k, q in zip(keys, partition) if q == p} for p in PARTITIONS}
    assert key_sets["train"].isdisjoint(key_sets["val"]), "train and val share keys"

    row_partition = partition[paired.group]
    rows = {p: np.flatnonzero(row_partition == p) for p in PARTITIONS}
    return Split(partition, rows)


def shared_keys(paired: PairedData, test: PairedData) -> list[tuple[float, str]]:
    """Raw keys present in both the training pair and the external test pair."""
    train_keys = {_key(r) for r in paired.groups}
    return sorted(k for k in (_key(r) for r in test.groups) if k in train_keys)


def split_manifest(
    paired: PairedData,
    split: Split,
    data_cfg: DataConfig,
    test: PairedData,
    checksums: dict[str, str],
    alpha: float = 0.0,
) -> dict:
    """Per-group partition record, partition counts, per-path weight shares, test files."""
    sizes = paired.group_sizes
    groups = [
        {
            "partition": p,
            "force": r["force"],
            "path": r["path"],
            "dataset": r["dataset"],
            "n_real": int(n),
            "source_rows": r["real_id"],
        }
        for r, p, n in zip(paired.groups, split.partition, sizes)
    ]
    counts: dict[str, dict[str, float]] = {
        p: {
            "groups": int((split.partition == p).sum()),
            "readings": int(sizes[split.partition == p].sum()),
        }
        for p in PARTITIONS
    }
    total = sum(c["readings"] for c in counts.values())
    for c in counts.values():
        c["reading_share"] = c["readings"] / total

    # effective share of the training loss per path under group weights n_g ** alpha,
    # against the share of readings (micro weighting)
    train = split.partition == "train"
    weight: Counter[str] = Counter()
    readings: Counter[str] = Counter()
    for r, n, t in zip(paired.groups, sizes, train):
        if t:
            weight[r["path"]] += float(n) ** alpha
            readings[r["path"]] += int(n)
    w_total, r_total = sum(weight.values()), sum(readings.values())
    path_share = {
        path: {
            "loss_weight_share": weight[path] / w_total,
            "reading_share": readings[path] / r_total,
        }
        for path in sorted(weight)
    }
    return {
        "version": MANIFEST_VERSION,
        "seed": data_cfg.split_seed,
        "config": {
            "val_ratio": data_cfg.val_ratio,
            "split_strategy": "group",
            "group_weight_alpha": alpha,
        },
        "counts": counts,
        "test": {
            "files": {f.name: checksums[f.name] for f in test.files},
            "key_cols": list(data_cfg.test_key_cols),
            "groups": len(test.groups),
            "readings": len(test.group),
        },
        "train_path_weight_share": path_share,
        "groups": groups,
    }
