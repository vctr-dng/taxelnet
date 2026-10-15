"""Few-shot gain/offset calibration protocol on trained run bundles (taxelnet.calibration).

For each bundle: rebuild the data and split from config.json, recompute raw-TC
predictions with model.pt (or best.pt) + preprocessing.json on the grasp test set and on
the held-out-shape val set (same-domain control), run the protocol per object group (test)
/ per held-out shape (val) and pooled, write calibration.json and log a summary to a new
MLflow run `p1-calib-<source run name>`.

Usage: python scripts/calibrate_eval.py BUNDLE [BUNDLE ...] [--out-root DIR]
       [--tracking-uri URI] [--experiment NAME] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from taxelnet import artifacts
from taxelnet.calibration import DEFAULT_TAU, VARIANTS, run_protocol, shots_to_oracle
from taxelnet.checkpoint import load_checkpoint, load_model
from taxelnet.config import Configs, ModelConfig
from taxelnet.dataset import TaxelDataset, load_paired, load_test_paired
from taxelnet.evaluation import object_group, predict
from taxelnet.model import build_model
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import make_split, shape_of

SUMMARY_METRICS = ("macro_mae", "cosine_median", "macro_aligned_mae", "gain", "offset")


def load_configs(bundle: Path) -> Configs:
    cfgs = Configs()
    for section, values in json.loads((bundle / "config.json").read_text()).items():
        cfg = getattr(cfgs, section)
        for k, v in values.items():
            if not hasattr(cfg, k):
                continue  # field of another branch: irrelevant for evaluation
            cur = getattr(cfg, k)
            if isinstance(v, list):
                v = tuple(v)
            elif isinstance(cur, Path) or k.endswith("_dir"):
                v = None if v is None else Path(v)
            setattr(cfg, k, v)
    return cfgs


def load_bundle_model(bundle: Path, cfgs: Configs, taxel_shape) -> tuple[Any, dict]:
    if (bundle / "model.pt").exists():
        model = load_model(bundle / "model.pt")
        ckpt = torch.load(bundle / "best.pt", map_location="cpu", weights_only=True)
        return model, {k: v for k, v in ckpt.items() if not k.endswith("_state")}
    model = build_model(ModelConfig(**vars(cfgs.model)), taxel_shape)
    ckpt = load_checkpoint(bundle / "best.pt", model)
    return model.eval(), {k: v for k, v in ckpt.items() if not k.endswith("_state")}


def domain_protocols(pred, target, group, domains, args) -> dict[str, Any]:
    out = {}
    for name, sel in domains.items():
        out[name] = run_protocol(
            pred[sel], target[sel], group[sel],
            ks=tuple(args.ks), n_draws=args.draws, tau=args.tau, seed=args.seed,
        )  # fmt: skip
    return out


def summarize(results: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Flat MLflow metrics and per-domain shots-to-oracle (10 %)."""
    flat: dict[str, float] = {}
    shots: dict[str, Any] = {}
    for split, domains in results.items():
        for dom, res in domains.items():
            tag = re.sub(r"[^A-Za-z0-9_.-]", "_", dom)
            for v in VARIANTS:
                for k, r in res[v].items():
                    for m in SUMMARY_METRICS:
                        key = m if k == "all" else f"{m}_mean"
                        if key in r:
                            flat[f"calib_{split}_{tag}_{v}_k{k}_{m}"] = r[key]
                    if k != "all":
                        flat[f"calib_{split}_{tag}_{v}_k{k}_macro_mae_std"] = r[
                            "macro_mae_std"
                        ]
                s = shots_to_oracle(res, v)
                shots[f"{split}/{dom}/{v}"] = s
                flat[f"calib_{split}_{tag}_{v}_shots_to_oracle10"] = float(
                    s if s is not None else -1
                )
    return flat, shots


def object_macro(results: dict[str, Any]) -> dict[str, Any]:
    """Mean over test object groups of the per-group calibrated macro MAE."""
    groups = [d for d in results if d != "pooled"]
    out: dict[str, Any] = {}
    for v in VARIANTS:
        out[v] = {}
        for k in results[groups[0]][v]:
            key = "macro_mae" if k == "all" else "macro_mae_mean"
            vals = [results[g][v][k][key] for g in groups if k in results[g][v]]
            if len(vals) == len(groups):
                out[v][k] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/vctr-dng/robotic/taxelnet/outputs/lab/calib"),
    )
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tracking-uri", default="http://192.168.193.150:5000")
    ap.add_argument("--experiment", default="taxelnet-v0-lab")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    for bundle in args.bundles:
        source_name = bundle.name.split("_", 1)[1] if "_" in bundle.name else bundle.name
        cfgs = load_configs(bundle)
        paired = load_paired(cfgs.data)
        test = load_test_paired(cfgs.data)
        split = make_split(paired, cfgs.data)
        pre = Preprocessor.load(bundle / "preprocessing.json")
        model, ckpt = load_bundle_model(bundle, cfgs, paired.taxel_shape)

        val_rows = split.rows["val"]
        test_rows = np.arange(len(test.group))
        preds, targets, groups = {}, {}, {}
        for p, (src, rows) in {"val": (paired, val_rows), "test": (test, test_rows)}.items():
            ds = TaxelDataset(src, rows, pre.transform_x, pre.transform_y)
            preds[p] = predict(model, ds, pre).reshape(len(rows), -1)
            targets[p] = src.y[rows].reshape(len(rows), -1).astype(np.float64)
            groups[p] = src.group[rows]

        check: dict[str, Any] = {}
        parquet = bundle / "predictions_test.parquet"
        if parquet.exists():
            import polars as pl

            ref = np.stack(pl.read_parquet(parquet)["prediction"].to_list())
            check["max_abs_diff_vs_predictions_test"] = float(
                np.abs(ref - preds["test"]).max()
            )

        names = np.array([object_group(test.groups[g]["path"]) for g in groups["test"]])
        shapes = np.array([shape_of(paired.groups[g]["path"]) for g in groups["val"]])
        test_domains = {"pooled": np.ones(len(names), bool)} | {
            o: names == o for o in sorted(set(names))
        }
        val_domains = {"pooled": np.ones(len(shapes), bool)} | {
            s: shapes == s for s in sorted(set(shapes))
        }
        results = {
            "test": domain_protocols(
                preds["test"], targets["test"], groups["test"], test_domains, args
            ),
            "val": domain_protocols(
                preds["val"], targets["val"], groups["val"], val_domains, args
            ),
        }
        flat, shots = summarize(results)
        obj_macro = object_macro(results["test"])
        for v, d in obj_macro.items():
            for k, val in d.items():
                flat[f"calib_test_objmacro_{v}_k{k}_macro_mae"] = val

        out_dir = args.out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{source_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_bundle": str(bundle),
            "source_run_name": source_name,
            "source_run_id": ckpt.get("mlflow_run_id"),
            "best_epoch": ckpt.get("best_epoch"),
            "protocol": {
                "ks": args.ks, "draws": args.draws, "tau": args.tau, "seed": args.seed,
                "unit": "raw (force, key) group; test keys are singletons",
                "fit": "L1: weighted median (gain), IRLS (affine), ridge prior "
                "toward the global affine fit worth tau readings (taxel_affine)",
                "oracle": "k=all: fit and evaluate on every reading of the domain",
            },  # fmt: skip
            "check": check,
            "test_object_macro_mae": obj_macro,
            "shots_to_oracle10": shots,
            "results": results,
        }
        artifacts.write_json(out_dir / "calibration.json", payload)
        print(f"{source_name}: {out_dir / 'calibration.json'} {check}")
        for v in VARIANTS:
            print(f"  objmacro {v:14s}", {k: round(x, 1) for k, x in obj_macro[v].items()})

        if args.no_mlflow:
            continue
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=f"p1-calib-{source_name}"):
            mlflow.log_params(
                {
                    "source_run_id": ckpt.get("mlflow_run_id"),
                    "source_run_name": source_name,
                    "source_bundle": str(bundle),
                    "calib.ks": ",".join(map(str, args.ks)),
                    "calib.draws": args.draws,
                    "calib.tau": args.tau,
                    "calib.seed": args.seed,
                }
            )
            mlflow.set_tags(
                {"git.commit": artifacts.git_revision(Path(__file__).parent)["commit"]}
            )
            mlflow.log_metrics(flat)
            mlflow.log_artifact(str(out_dir / "calibration.json"))


if __name__ == "__main__":
    main()
