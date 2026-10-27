"""Write the fixed dataset split for the Pushing_task leakage experiment.

Rows are the rows of data/raw/Dataset_training_real.csv (one training sample each). Groups
are (force rounded to 2 decimals, path). Partitions:
  train, val, test_clean: whole groups (seeded, as close to 80/10/10 of the rows as possible)
  test_leaky: rows carved out of multi-row *train* groups, so each one shares its
              (path, force) key with rows that stay in train. Same size as test_clean.
Train, val and test_clean never share a key; test_leaky is excluded from train.

Usage: uv run python scripts/make_pushing_split.py [--seed 0] [--out data/splits/pushing_split.json]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from taxelnet.splits import allocate_groups

FORCE_DECIMALS = 2
RATIOS = {"train": 0.8, "val": 0.1, "test_clean": 0.1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/splits/pushing_split.json"))
    args = ap.parse_args()

    real_path = args.raw_dir / "Dataset_training_real.csv"
    real = pl.read_csv(real_path)
    force = real["force"].round(FORCE_DECIMALS).to_list()
    path = real["path"].cast(pl.Utf8).str.strip_chars().to_list()
    keys = list(zip(path, force))

    uniq = sorted(set(keys))
    index = {k: i for i, k in enumerate(uniq)}
    group = np.array([index[k] for k in keys])
    sizes = np.bincount(group, minlength=len(uniq))
    part = allocate_groups(sizes, RATIOS, args.seed)
    row_part = part[group]
    rows = {p: np.flatnonzero(row_part == p) for p in RATIOS}

    # leaky test: round-robin over shuffled multi-row train groups, one row per pass,
    # always leaving at least one row of the group in train
    rng = np.random.default_rng(args.seed)
    n_leaky = len(rows["test_clean"])
    pools = {
        g: rng.permutation(np.flatnonzero((group == g) & (row_part == "train")))
        for g in rng.permutation(np.flatnonzero((sizes >= 2) & (part == "train")))
    }
    leaky: list[int] = []
    while len(leaky) < n_leaky:
        progressed = False
        for g, pool in pools.items():
            if len(leaky) < n_leaky and len(pool) > 1:
                leaky.append(int(pool[-1]))
                pools[g] = pool[:-1]
                progressed = True
        if not progressed:
            raise RuntimeError("not enough multi-row train groups for the leaky test")
    leaky_rows = np.array(sorted(leaky))
    rows["train"] = np.setdiff1d(rows["train"], leaky_rows)
    rows["test_leaky"] = leaky_rows

    # invariants
    keyset = {p: {keys[i] for i in r} for p, r in rows.items()}
    assert keyset["train"].isdisjoint(keyset["val"])
    assert keyset["train"].isdisjoint(keyset["test_clean"])
    assert keyset["val"].isdisjoint(keyset["test_clean"])
    assert keyset["test_leaky"] <= keyset["train"], "leaky test keys must appear in train"
    assert len({i for r in rows.values() for i in r}) == sum(len(r) for r in rows.values())

    n = len(keys)
    out = {
        "description": __doc__.strip().split("\n\n")[0],
        "seed": args.seed,
        "real_csv": real_path.name,
        "real_csv_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "n_rows": n,
        "counts": {
            p: {
                "rows": len(r),
                "row_share": round(len(r) / n, 5),
                "groups": len(keyset[p]),
            }
            for p, r in rows.items()
        },
        "leaky_test_rows_with_key_in_train": int(
            sum(keys[i] in keyset["train"] for i in rows["test_leaky"])
        ),
        "rows": {p: r.tolist() for p, r in rows.items()},  # indices into the real CSV
        "groups": {
            p: sorted([k[0], k[1]] for k in ks) for p, ks in keyset.items()
        },  # [path, force]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ("seed", "n_rows", "counts")}, indent=1))
    print("leaky rows with key in train:", out["leaky_test_rows_with_key_in_train"])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
