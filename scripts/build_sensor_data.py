"""Build the grasp-only training pair (with a `sensor` column) from the grasp CSVs.

Usage: python scripts/build_sensor_data.py --out-dir DIR [--drop-above TC] [--data-dir DIR]

Reads Dataset_Grasp_{Real,Simulation}_qtac100.csv (force, Name, ...) and writes
Dataset_GraspOnly_{Real,Simulation}.csv in the loader's training format:
  Real        force, Path, dataset, sensor, tax1 ... tax28   (std* dropped)
  Simulation  force, Path, dataset, n195 ... n14899
`Name` `Cuboid_G_20_02_A_V+1_H+1_Deg15_S1` becomes Path `Cuboid_G/20_02_A_V+1_H+1_Deg15_S1`
(object before "/", so split_strategy=shape holds out whole objects with both pads) and
sensor `S1` (last Name token). The loader pairs the pad label from the Path suffix; the real
file's `sensor` column is provenance and is checked against it here. `--drop-above TC`
drops, in both files, every row whose real file has any taxel above TC (glitch rows).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path("/home/vctr-dng/robotic/taxelnet/data/newton/v1/training")
REAL, SIM = "Dataset_Grasp_Real_qtac100.csv", "Dataset_Grasp_Simulation_qtac100.csv"
OUT_REAL, OUT_SIM = "Dataset_GraspOnly_Real.csv", "Dataset_GraspOnly_Simulation.csv"
N_TAX = 28


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--drop-above", type=float, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.data_dir / REAL).open() as f:
        real_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    with (args.data_dir / SIM).open() as f:
        sim_lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    rh, sh = real_lines[0].split(","), sim_lines[0].split(",")
    if rh[:2] != ["force", "Name"] or sh[:2] != ["force", "Name"]:
        raise ValueError("unexpected grasp headers")
    if len(real_lines) != len(sim_lines):
        raise ValueError("real and simulation row counts differ")
    taxels = rh[2 : 2 + N_TAX]

    kept, dropped, counts = [], 0, {}
    out_real = [",".join(["force", "Path", "dataset", "sensor", *taxels])]
    out_sim = [",".join(["force", "Path", "dataset", *sh[2:]])]
    for rl, sl in zip(real_lines[1:], sim_lines[1:]):
        r, s = rl.split(","), sl.split(",")
        if r[:2] != s[:2]:
            raise ValueError(f"row key mismatch: {r[:2]} vs {s[:2]}")
        name = r[1]
        shape, size, rest = name.split("_", 2)
        sensor = name.rsplit("_", 1)[1]
        if sensor not in ("S1", "S2"):
            raise ValueError(f"unexpected sensor token in {name}")
        if args.drop_above is not None and max(
            float(v) for v in r[2 : 2 + N_TAX]
        ) > args.drop_above:
            dropped += 1
            continue
        path = f"{shape}_{size}/{rest}"
        out_real.append(",".join([r[0], path, "grasp", sensor, *r[2 : 2 + N_TAX]]))
        out_sim.append(",".join([s[0], path, "grasp", *s[2:]]))
        counts[sensor] = counts.get(sensor, 0) + 1
        kept.append(name)

    (args.out_dir / OUT_REAL).write_text("\n".join(out_real) + "\n")
    (args.out_dir / OUT_SIM).write_text("\n".join(out_sim) + "\n")
    manifest = {"drop_above": args.drop_above, "rows": len(kept), "dropped": dropped, "per_sensor": counts}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(manifest)


if __name__ == "__main__":
    main()
