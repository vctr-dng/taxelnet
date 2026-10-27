"""Merge the indenter and grasp training pairs into one matched pair the loaders accept.

Usage: uv run python scripts/merge_training_data.py [--data-dir DIR] [--out-dir DIR]

Inputs (in --data-dir, default data/newton/v1/training):
  Dataset_Indenter_Real.csv / Dataset_Indenter_Simulation.csv   force, Path, dataset, ...
  Dataset_Grasp_Real_qtac100.csv / Dataset_Grasp_Simulation_qtac100.csv
                                                                force, Name, ...

Outputs (in --out-dir, default data/newton/v1/merged):
  Dataset_Merged_Real.csv          force, Path, dataset, tax1 ... tax28
  Dataset_Merged_Simulation.csv    force, Path, dataset, n195 ... n14899
  merge_manifest.json              source checksums, row ranges, grasp Path mapping

Indenter rows are copied verbatim. A grasp row `Name` such as
`Cuboid_G_20_02_A_V+1_H+1_Deg15_S1` becomes Path `Cuboid_G/20_02_A_V+1_H+1_Deg15_S1`
(object before "/", so `split_strategy=shape` holds out whole grasp objects with both
pads S1/S2) and dataset `grasp`. The grasp real `std1 ... std28` columns are dropped.
Values are copied as text, never re-formatted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taxelnet.io_csv import file_sha256, sim_node_columns

INDENTER = ("Dataset_Indenter_Real.csv", "Dataset_Indenter_Simulation.csv")
GRASP = ("Dataset_Grasp_Real_qtac100.csv", "Dataset_Grasp_Simulation_qtac100.csv")
MERGED = ("Dataset_Merged_Real.csv", "Dataset_Merged_Simulation.csv")
TAXELS = [f"tax{i}" for i in range(1, 29)]
GRASP_DATASET = "grasp"


def grasp_path(name: str) -> str:
    """`<Shape>_<Size>_<rest>` -> `<Shape>_<Size>/<rest>`."""
    shape, size, rest = name.split("_", 2)
    return f"{shape}_{size}/{rest}"


def merge_pair(
    indenter_csv: Path, grasp_csv: Path, out_csv: Path, value_cols: list[str]
) -> dict:
    """Write indenter rows then grasp rows; returns row counts and grasp object names."""
    with indenter_csv.open() as f_ind, grasp_csv.open() as f_gr, out_csv.open("w") as out:
        ind_header = f_ind.readline().rstrip("\n").split(",")
        gr_header = f_gr.readline().rstrip("\n").split(",")
        if ind_header != ["force", "Path", "dataset", *value_cols]:
            raise ValueError(f"{indenter_csv.name}: unexpected header {ind_header[:5]}...")
        if gr_header[:2] != ["force", "Name"] or gr_header[2 : 2 + len(value_cols)] != value_cols:
            raise ValueError(f"{grasp_csv.name}: unexpected header {gr_header[:5]}...")
        n_keep = 2 + len(value_cols)  # drops trailing std* columns of the real file
        out.write(",".join(ind_header) + "\n")

        n_ind = 0
        ind_shapes: set[str] = set()
        for line in f_ind:
            if line.strip():
                out.write(line if line.endswith("\n") else line + "\n")
                ind_shapes.add(line.split(",", 2)[1].split("/", 1)[0])
                n_ind += 1

        n_gr = 0
        objects: set[str] = set()
        for line in f_gr:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split(",")
            if len(fields) < n_keep:
                raise ValueError(f"{grasp_csv.name}: short row {n_gr + 2}")
            path = grasp_path(fields[1])
            objects.add(path.split("/", 1)[0])
            out.write(",".join([fields[0], path, GRASP_DATASET, *fields[2:n_keep]]) + "\n")
            n_gr += 1

    clash = ind_shapes & objects
    if clash:
        raise ValueError(f"grasp objects collide with indenter shapes: {sorted(clash)}")
    return {"indenter_rows": n_ind, "grasp_rows": n_gr, "grasp_objects": sorted(objects)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/newton/v1/training"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/newton/v1/merged"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    value_cols = (TAXELS, sim_node_columns())
    report = {}
    for kind, ind, gr, out, cols in zip(("real", "sim"), INDENTER, GRASP, MERGED, value_cols):
        print(f"merging {ind} + {gr} -> {out}", flush=True)
        report[kind] = merge_pair(args.data_dir / ind, args.data_dir / gr, args.out_dir / out, cols)
    if report["real"] != report["sim"]:
        raise ValueError(f"real and simulation merges differ: {report}")

    manifest = {
        "sources": {
            name: file_sha256(args.data_dir / name) for name in (*INDENTER, *GRASP)
        },
        "outputs": {name: file_sha256(args.out_dir / name) for name in MERGED},
        "rows": {
            "indenter": [0, report["real"]["indenter_rows"]],
            "grasp": [
                report["real"]["indenter_rows"],
                report["real"]["indenter_rows"] + report["real"]["grasp_rows"],
            ],
        },
        "grasp_mapping": "Path = <Shape>_<Size>/<rest of Name>; dataset = 'grasp'; std* dropped",
        "grasp_objects": report["real"]["grasp_objects"],
    }
    (args.out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["rows"]), f"{len(manifest['grasp_objects'])} grasp objects")


if __name__ == "__main__":
    main()
