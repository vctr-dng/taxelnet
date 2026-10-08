"""Seeded, weighted group-level train/val/test allocation and split manifests.

A full raw (force, Path) group is indivisible: all its presses go to one partition. Group
weight is its number of real readings. The test partition is persisted once as an immutable
manifest and reused by every later run, so regression gates compare runs on one test set.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from taxelnet.config import DataConfig
from taxelnet.dataset import PairedData
from taxelnet.io_csv import MAX_REPORTED

PARTITIONS = ("train", "val", "test")
MANIFEST_VERSION = 1


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
    test_manifest_path: Path
    test_manifest_created: bool


def make_split(paired: PairedData, data_cfg: DataConfig, real_sha256: str) -> Split:
    """Train/val/test per group, reusing (or creating) the immutable test manifest."""
    sizes = paired.group_sizes
    keys = [_key(r) for r in paired.groups]
    key_to_group = {k: g for g, k in enumerate(keys)}
    path = data_cfg.test_manifest

    if path.exists():
        manifest = json.loads(path.read_text())
        test_keys = [(float(f), p) for f, p in manifest["keys"]]
        missing = [k for k in test_keys if k not in key_to_group]
        if missing:
            raise ValueError(
                f"{path}: {len(missing)} test keys are not in the data, "
                f"e.g. {missing[:MAX_REPORTED]}"
            )
        if manifest.get("real_csv_sha256") != real_sha256:
            raise ValueError(
                f"{path} was made from a different real CSV "
                f"({manifest.get('real_csv_sha256')} != {real_sha256}); runs on different "
                "test manifests are not comparable. Move it aside to start a new test set."
            )
        partition = np.full(len(keys), None, dtype=object)
        test_groups = np.array([key_to_group[k] for k in test_keys], dtype=np.int64)
        partition[test_groups] = "test"
        rest = np.flatnonzero([p is None for p in partition])
        val_share = data_cfg.val_ratio / (1 - data_cfg.test_ratio)
        partition[rest] = allocate_groups(
            sizes[rest], {"train": 1 - val_share, "val": val_share}, data_cfg.split_seed
        )
        created = False
    else:
        partition = allocate_groups(
            sizes,
            {
                "train": 1 - data_cfg.val_ratio - data_cfg.test_ratio,
                "val": data_cfg.val_ratio,
                "test": data_cfg.test_ratio,
            },
            data_cfg.split_seed,
        )
        test_keys = sorted(k for k, p in zip(keys, partition) if p == "test")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "version": MANIFEST_VERSION,
                    "description": "Immutable Taxelnet test split: raw (force, Path) keys",
                    "real_csv": data_cfg.real_csv,
                    "real_csv_sha256": real_sha256,
                    "seed": data_cfg.split_seed,
                    "test_ratio": data_cfg.test_ratio,
                    "n_groups": len(test_keys),
                    "n_readings": int(sizes[partition == "test"].sum()),
                    "keys": [[f, p] for f, p in test_keys],
                },
                indent=1,
            )
        )
        created = True

    key_sets = {p: {k for k, q in zip(keys, partition) if q == p} for p in PARTITIONS}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1 :]:
            assert key_sets[a].isdisjoint(key_sets[b]), f"{a} and {b} share keys"

    row_partition = partition[paired.group]
    rows = {p: np.flatnonzero(row_partition == p) for p in PARTITIONS}
    return Split(partition, rows, path, created)


def split_manifest(
    paired: PairedData, split: Split, data_cfg: DataConfig, alpha: float = 0.0
) -> dict:
    """Per-group partition record, partition counts, and per-path weight shares."""
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
            "test_ratio": data_cfg.test_ratio,
            "split_strategy": "group",
            "test_manifest": str(split.test_manifest_path),
            "test_manifest_created": split.test_manifest_created,
            "group_weight_alpha": alpha,
        },
        "counts": counts,
        "train_path_weight_share": path_share,
        "groups": groups,
    }
