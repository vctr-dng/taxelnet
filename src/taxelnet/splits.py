"""Seeded, weighted group-level train/val allocation and split manifests.

A full raw (force, Path) group is indivisible: all its presses go to one partition. Group
weight is its number of real readings. The test set is not carved from the training data:
it is the external pair in `DataConfig.test_dir`, identical for every run.

`DataConfig.split_strategy`:
- "group": (force, Path) groups are allocated independently (val = force interpolation
  on seen shapes).
- "shape": whole indenter shapes (the part of `Path` before "/") are held out, so val
  measures generalization to unseen shapes. No shape crosses partitions.
- "shape_fold": k-fold over shapes (`n_folds`, `fold`): shapes are dealt into folds
  balanced by reading count (`shape_folds`), fold `fold` is val, the rest is train.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from taxelnet.config import DataConfig
from taxelnet.dataset import PairedData

PARTITIONS = ("train", "val")  # carved from the training pair
TEST = "test"  # the external test pair
MANIFEST_VERSION = 3


def allocate_groups(
    group_sizes: np.ndarray,
    ratios: dict[str, float],
    seed: int,
    largest_first: bool = True,
) -> np.ndarray:
    """Partition name per group, with reading counts as close as possible to `ratios`.

    Groups are shuffled (seeded), stable-sorted by size, largest first, then each one goes
    to the partition with the largest remaining deficit relative to its target reading
    count. The relative deficit spreads large repeated groups over every partition in
    proportion, instead of filling the largest partition with them first.

    `largest_first=False` (for a few large units, e.g. whole shapes) keeps the seeded
    random order and sends each unit only to partitions it does not overfill (falling
    back to the least relative overfill when it fits nowhere). Size-sorting there would
    put the same large unit in val for every seed.
    """
    group_sizes = np.asarray(group_sizes)
    names = list(ratios)
    total = group_sizes.sum()
    targets = np.array([ratios[n] for n in names], dtype=float)
    targets = targets / targets.sum() * total
    counts = np.zeros(len(names))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(group_sizes))
    order = (
        perm[np.argsort(-group_sizes[perm], kind="stable")] if largest_first else perm
    )

    out = np.empty(len(group_sizes), dtype=object)
    for g in order:
        deficit = (targets - counts) / targets
        if not largest_first:
            after = (targets - counts - group_sizes[g]) / targets
            # among partitions it fits in: largest deficit; else: least relative overfill
            deficit = (
                np.where(after >= 0, deficit, -np.inf) if (after >= 0).any() else after
            )
        p = int(np.argmax(deficit))
        out[g] = names[p]
        counts[p] += group_sizes[g]
    return out


SPLIT_STRATEGIES = ("group", "shape", "shape_fold")


def shape_folds(shape_sizes: np.ndarray, n_folds: int) -> np.ndarray:
    """Fold index per shape: largest shape first, each to the fold with fewest readings.

    Deterministic (no seed): every fold of a k-fold run sees the same assignment. Ties
    go to the lowest fold index.
    """
    if n_folds < 2 or n_folds > len(shape_sizes):
        raise ValueError(f"n_folds must be in [2, {len(shape_sizes)}], got {n_folds}")
    fold = np.empty(len(shape_sizes), dtype=np.int64)
    load = np.zeros(n_folds)
    for s in np.argsort(-np.asarray(shape_sizes), kind="stable"):
        f = int(np.argmin(load))
        fold[s] = f
        load[f] += shape_sizes[s]
    return fold


def _key(record: dict) -> tuple[float, str]:
    return (float(record["force"]), record["path"])


def shape_of(path: str) -> str:
    """Indenter shape of a training `Path`: the part before the first "/"."""
    return path.split("/", 1)[0]


@dataclass(frozen=True)
class Split:
    partition: np.ndarray  # partition name per group
    rows: dict[str, np.ndarray]  # sorted data rows per partition
    strategy: str = "group"
    held_out_shapes: tuple[str, ...] = ()  # val shapes ("shape"/"shape_fold" only)


def make_split(paired: PairedData, data_cfg: DataConfig) -> Split:
    """Seeded train/val allocation of the training groups (no test carve-out)."""
    keys = [_key(r) for r in paired.groups]
    ratios = {"train": 1 - data_cfg.val_ratio, "val": data_cfg.val_ratio}
    strategy = data_cfg.split_strategy
    held_out: tuple[str, ...] = ()
    if strategy == "group":
        partition = allocate_groups(paired.group_sizes, ratios, data_cfg.split_seed)
    elif strategy == "shape":
        shapes = np.array([shape_of(r["path"]) for r in paired.groups])
        names, inverse = np.unique(shapes, return_inverse=True)
        shape_sizes = np.bincount(inverse, weights=paired.group_sizes).astype(int)
        shape_part = allocate_groups(
            shape_sizes, ratios, data_cfg.split_seed, largest_first=False
        )
        partition = shape_part[inverse]
        held_out = tuple(sorted(names[shape_part == "val"].tolist()))
        assert set(shapes[partition == "train"]).isdisjoint(
            shapes[partition == "val"]
        ), "train and val share shapes"
    elif strategy == "shape_fold":
        shapes = np.array([shape_of(r["path"]) for r in paired.groups])
        names, inverse = np.unique(shapes, return_inverse=True)
        shape_sizes = np.bincount(inverse, weights=paired.group_sizes).astype(int)
        fold_of = shape_folds(shape_sizes, data_cfg.n_folds)
        if not 0 <= data_cfg.fold < data_cfg.n_folds:
            raise ValueError(f"fold {data_cfg.fold} not in [0, {data_cfg.n_folds})")
        is_val = fold_of == data_cfg.fold
        partition = np.where(is_val, "val", "train").astype(object)[inverse]
        held_out = tuple(sorted(names[is_val].tolist()))
    else:
        raise ValueError(
            f"unknown split_strategy {strategy!r}, expected one of {SPLIT_STRATEGIES}"
        )
    key_sets = {p: {k for k, q in zip(keys, partition) if q == p} for p in PARTITIONS}
    assert key_sets["train"].isdisjoint(key_sets["val"]), "train and val share keys"

    row_partition = partition[paired.group]
    rows = {p: np.flatnonzero(row_partition == p) for p in PARTITIONS}
    return Split(partition, rows, strategy, held_out)


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
            "shapes": len(
                {
                    shape_of(r["path"])
                    for r, q in zip(paired.groups, split.partition)
                    if q == p
                }
            ),
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
            "split_strategy": split.strategy,
            "group_weight_alpha": alpha,
            **(
                {"n_folds": data_cfg.n_folds, "fold": data_cfg.fold}
                if split.strategy == "shape_fold"
                else {}
            ),
        },
        "held_out_shapes": list(split.held_out_shapes),
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
