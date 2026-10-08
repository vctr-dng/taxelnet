"""Dataset fingerprints, JSON metadata, and model-bundle files."""

from __future__ import annotations

import csv
import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from taxelnet.dataset import PairedData
from taxelnet.io_csv import SIM_GRID_SHAPE, sim_node_columns

KEY_PACKAGES = ("taxelnet", "torch", "numpy", "polars", "mlflow")

# Real presses left out of the matched export, as reported by Dataset_Indenter_README.pdf.
# Unmatched_real.txt, which lists them, is not in data/Indenters.
README_EXCLUSIONS = {
    "source": "Dataset_Indenter_README.pdf; Unmatched_real.txt not available",
    "total": 688,
    "circlexs_above_10N_not_simulated": 601,
    "invalid_simulation": 83,
    "force_le_0": 4,
    "flat_paths": "absent from the export (count not reported)",
}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1))


def write_node_ids_csv(path: Path) -> None:
    """Row-major 77 x 49 node ids, in the order of io_csv.sim_node_columns()."""
    n_cols = SIM_GRID_SHAPE[1]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "node_id"])
        for k, name in enumerate(sim_node_columns()):
            writer.writerow([k // n_cols, k % n_cols, int(name[1:])])


def package_versions() -> dict[str, str]:
    out = {}
    for name in KEY_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_requirements_lock(path: Path) -> None:
    dists = sorted(
        {d.metadata["Name"]: d.version for d in metadata.distributions()}.items(),
        key=lambda kv: kv[0].lower(),
    )
    path.write_text("".join(f"{name}=={version}\n" for name, version in dists))


def git_revision(cwd: Path | None = None) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def schema_report(paired: PairedData, checksums: dict[str, str]) -> dict[str, Any]:
    sizes = paired.group_sizes
    repeated = sizes >= 2
    paths = {g["path"] for g in paired.groups}
    return {
        "files": checksums,
        "n_rows": len(paired.group),
        "n_keys": len(paired.groups),
        "n_paths": len(paths),
        "n_singleton_keys": int((sizes == 1).sum()),
        "n_repeated_keys": int(repeated.sum()),
        "n_readings_in_repeated_keys": int(sizes[repeated].sum()),
        "max_group_size": int(sizes.max()),
        "n_keys_with_differing_sim_rows": paired.n_sim_differ,
        "datasets": sorted({g["dataset"] for g in paired.groups}),
        "sim": {
            "grid_shape": list(paired.mesh_shape),
            "units": "mm",
            "min": float(paired.x.min()),
            "max": float(paired.x.max()),
            "negative_share": float((paired.x < 0).mean()),
        },
        "real": {
            "grid_shape": list(paired.taxel_shape),
            "units": "TC",
            "min": float(paired.y.min()),
            "max": float(paired.y.max()),
            "negative_share": float((paired.y < 0).mean()),
        },
        "force": {
            "min": float(min(g["force"] for g in paired.groups)),
            "max": float(max(g["force"] for g in paired.groups)),
        },
        "excluded": README_EXCLUSIONS,
    }
