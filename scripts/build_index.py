"""Build <data-dir>/index.json: one record per (force, Path) key pair (see taxelnet.dataset.build_index).

Usage: uv run python scripts/build_index.py [--data-dir data/newton/v1/training]
"""

import argparse
import json
from pathlib import Path

from taxelnet.dataset import build_index
from taxelnet.io_csv import REAL_CSV, SIM_CSV

DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "newton" / "v1" / "training"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    index, n_sim_differ = build_index(args.data_dir / REAL_CSV, args.data_dir / SIM_CSV)
    out = args.data_dir / "index.json"
    out.write_text(json.dumps(index, indent=1))
    n_repeat = sum(len(e["real_id"]) > 1 for e in index)
    print(
        f"{len(index)} key pairs ({n_repeat} with several real rows, "
        f"{n_sim_differ} with differing simulation rows) -> {out}"
    )


if __name__ == "__main__":
    main()
