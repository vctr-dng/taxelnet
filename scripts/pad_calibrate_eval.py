"""Per-test-pad gain calibration of trained bundles: the test pads are not the training pads.

The network output is converted to TC with the training scale of the row's pad label,
then each *test* pad gets its own correction fitted on a few of its own readings
(taxelnet.calibration, L1 fits). Protocols, all scored in raw TC on readings not used
for the fit:

- ``none``: training scales as labelled, no correction (reference only).
- ``pad_k<k>``: per pad, `k` random readings of that pad fit the correction; scored on
  every other test reading; `--draws` draws (same draws for every variant and bundle).
- ``s1_xobj``: per S1 test object, the S1 correction is fitted on all readings of the
  other S1 objects (a new object on a calibrated pad). The S2 pad has a single test
  object, so S2 uses its pad_k10 draws.
- ``pad_oracle``: fitted on every reading of the pad (optimistic: uses all test labels).

Variants: ``global_gain`` (y = g p) and ``global_affine`` (y = g p + o), one per pad.
Writes <bundle>/pad_calibration.json; with --summary, an arm x protocol table over the
bundles (arm = run name without the trailing -s<seed>).

Usage: python scripts/pad_calibrate_eval.py BUNDLE [BUNDLE ...] [--summary OUT_DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_eval import load_bundle_model, load_configs

from taxelnet import artifacts
from taxelnet.calibration import DEFAULT_TAU, eval_metrics, fit_calibration
from taxelnet.dataset import TaxelDataset, load_paired, load_test_paired
from taxelnet.evaluation import global_ssim, object_group, predict
from taxelnet.preprocessing import Preprocessor
from taxelnet.splits import make_split

VARIANTS = ("global_gain", "global_affine")
OUT_NAME = "pad_calibration.json"
TAU = DEFAULT_TAU  # taxel_affine: prior weight toward the pad global affine fit
KS = (1, 3, 5, 10, 20)
METRICS = ("macro_mae", "obj_mean_mae", "ssim_mean", "cosine_median")


def score(
    pred: np.ndarray, target: np.ndarray, group: np.ndarray, objects: np.ndarray,
    keep: np.ndarray, data_range: float,
) -> dict[str, float]:  # fmt: skip
    """Metrics on the `keep` readings: overall, per object, mean over objects."""
    p, t = pred[keep], target[keep]
    m = eval_metrics(p, t, group[keep])
    out = {k: m[k] for k in ("macro_mae", "cosine_median", "macro_aligned_mae")}
    ssim = global_ssim(p, t, data_range)
    out["ssim_mean"] = float(ssim.mean())
    err = np.abs(p - t).reshape(len(p), -1).mean(1)
    obj = objects[keep]
    per = {o: float(err[obj == o].mean()) for o in sorted(set(obj))}
    out |= {f"obj_{o}_mae": v for o, v in per.items()}
    out |= {f"obj_{o}_ssim": float(ssim[obj == o].mean()) for o in per}
    out["obj_mean_mae"] = float(np.mean(list(per.values())))
    return out


def apply_pad_fits(
    pred: np.ndarray, target: np.ndarray, pads: np.ndarray, fit_rows: dict[str, np.ndarray],
    variant: str, tau: float = DEFAULT_TAU,
) -> tuple[np.ndarray, dict[str, float]]:  # fmt: skip
    """Each pad's readings corrected with the fit on that pad's `fit_rows`."""
    out = pred.astype(np.float64).copy()
    gains = {}
    for pad, rows in fit_rows.items():
        c = fit_calibration(variant, pred[rows], target[rows], tau)
        sel = pads == pad
        out[sel] = c.apply(pred[sel].reshape(sel.sum(), -1)).reshape(out[sel].shape)
        # per-taxel variants: the median over taxels
        gains[f"gain_{pad}"] = float(np.median(c.gain))
        gains[f"offset_{pad}"] = float(np.median(c.offset))
    return out, gains


def mean_std(rows: list[dict[str, float]]) -> dict[str, float]:
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def evaluate_bundle(bundle: Path, draws: int, seed: int) -> dict[str, Any]:
    cfgs = load_configs(bundle)
    paired = load_paired(cfgs.data)
    test = load_test_paired(cfgs.data)
    pre = Preprocessor.load(bundle / "preprocessing.json")
    model, ckpt = load_bundle_model(bundle, cfgs, paired.taxel_shape)
    rows = np.arange(len(test.group))
    pred = predict(
        model, TaxelDataset(test, rows, pre.transform_x, pre.transform_y), pre
    )
    pred = pred.reshape(len(rows), -1).astype(np.float64)
    target = test.y.reshape(len(rows), -1).astype(np.float64)
    group, pads = test.group, np.asarray(test.sensor).astype(str)
    objects = np.array([object_group(test.groups[g]["path"]) for g in group])
    dr = cfgs.eval.ssim_data_range
    all_rows = np.ones(len(rows), bool)
    res: dict[str, Any] = {"none": score(pred, target, group, objects, all_rows, dr)}

    rng = np.random.default_rng(seed)
    pad_rows = {p: np.flatnonzero(pads == p) for p in sorted(set(pads))}
    picks = {
        k: [{p: rng.choice(r, k, replace=False) for p, r in pad_rows.items()}
            for _ in range(draws)]
        for k in KS
    }  # fmt: skip
    for v in VARIANTS:
        for k, draw_list in picks.items():
            runs = []
            for fit_rows in draw_list:
                cal, gains = apply_pad_fits(pred, target, pads, fit_rows, v, TAU)
                keep = all_rows.copy()
                keep[np.concatenate(list(fit_rows.values()))] = False
                runs.append(score(cal, target, group, objects, keep, dr) | gains)
            res[f"pad_k{k}/{v}"] = mean_std(runs)
        cal, gains = apply_pad_fits(pred, target, pads, pad_rows, v, TAU)
        res[f"pad_oracle/{v}"] = (
            score(cal, target, group, objects, all_rows, dr) | gains
        )

        # S1: leave one S1 object out; S2 (single object): its pad_k10 draws
        s1_obj = sorted(set(objects[pads == "S1"]))
        per_obj, ssim_obj = {}, {}
        for o in s1_obj:
            fit_rows = np.flatnonzero((pads == "S1") & (objects != o))
            c = fit_calibration(v, pred[fit_rows], target[fit_rows], TAU)
            sel = objects == o
            cal = c.apply(pred[sel])
            per_obj[o] = float(np.abs(cal - target[sel]).mean())
            ssim_obj[o] = float(global_ssim(cal, target[sel], dr).mean())
        s2 = res[f"pad_k10/{v}"]
        s2_obj = sorted(set(objects[pads == "S2"]))
        per_obj |= {o: s2[f"obj_{o}_mae"] for o in s2_obj}
        ssim_obj |= {o: s2[f"obj_{o}_ssim"] for o in s2_obj}
        res[f"s1_xobj/{v}"] = (
            {f"obj_{o}_mae": x for o, x in per_obj.items()}
            | {f"obj_{o}_ssim": x for o, x in ssim_obj.items()}
            | {"obj_mean_mae": float(np.mean(list(per_obj.values())))}
        )

    # validation: training pads, so the training scales apply as labelled
    split = make_split(paired, cfgs.data)
    vrows = split.rows["val"]
    vpred = predict(
        model, TaxelDataset(paired, vrows, pre.transform_x, pre.transform_y), pre
    )
    vt = paired.y[vrows]
    verr = np.abs(vpred - vt).reshape(len(vrows), -1).mean(1)
    vpads = np.asarray(paired.sensor[vrows]).astype(str)
    val = {"mae": float(verr.mean())} | {
        f"mae_{p}": float(verr[vpads == p].mean()) for p in sorted(set(vpads))
    }
    val["ssim_mean"] = float(global_ssim(vpred, vt, dr).mean())
    payload = {
        "source_bundle": str(bundle),
        "best_epoch": ckpt.get("best_epoch"),
        "protocol": {
            "ks": KS,
            "draws": draws,
            "seed": seed,
            "variants": VARIANTS,
            "tau": TAU,
        },
        "val": val,
        "test": res,
    }
    artifacts.write_json(bundle / OUT_NAME, payload)
    return payload


def arm_of(bundle: Path) -> str:
    name = bundle.name.split("_", 1)[1] if "_" in bundle.name else bundle.name
    return re.sub(r"-s\d+$", "", name)


def summarize(payloads: dict[Path, dict], out_dir: Path) -> None:
    by_arm: dict[str, list[dict]] = {}
    for b, p in payloads.items():
        by_arm.setdefault(arm_of(b), []).append(p)
    table: dict[str, Any] = {}
    for arm, ps in sorted(by_arm.items()):
        table[arm] = {"n": len(ps), "val": mean_std([p["val"] for p in ps])}
        for proto in ps[0]["test"]:
            rows = [
                {k: v for k, v in p["test"][proto].items() if not k.endswith("_std")}
                for p in ps
            ]
            table[arm][proto] = mean_std(rows)
    artifacts.write_json(out_dir / "results.json", table)

    objs = sorted(
        k[4:-4] for k in next(iter(table.values()))["none"] if re.match(r"obj_.*_mae$", k)
        and k != "obj_mean_mae"
    )  # fmt: skip

    def cell(d: dict, k: str) -> str:
        return f"{d[k]:.1f} ± {d[k + '_std']:.1f}" if k in d else "–"

    lines = []
    for v in VARIANTS:
        lines += [
            f"### Test, {v} (MAE TC, mean ± std over seeds)",
            "",
            "| arm | protocol | test MAE | 4-obj mean | "
            + " | ".join(objs) + " | SSIM | gain S1 | gain S2 |",
            "|---|---|---|---|" + "---|" * len(objs) + "---|---|---|",
        ]  # fmt: skip
        protos = [
            "none",
            *[f"pad_k{k}/{v}" for k in KS],
            f"s1_xobj/{v}",
            f"pad_oracle/{v}",
        ]
        for arm, d in table.items():
            for pr in protos:
                r = d[pr]
                ssim = f"{r['ssim_mean']:.3f}" if "ssim_mean" in r else "–"
                gs = [f"{r[g]:.2f}" if g in r else "–" for g in ("gain_S1", "gain_S2")]
                lines.append(
                    f"| {arm} | {pr.split('/')[0]} | {cell(r, 'macro_mae')} | "
                    f"{cell(r, 'obj_mean_mae')} | "
                    + " | ".join(cell(r, f"obj_{o}_mae") for o in objs)
                    + f" | {ssim} | {gs[0]} | {gs[1]} |"
                )
        lines.append("")
    lines += ["### Validation (training pads, as labelled)", "",
              "| arm | n | val MAE | S1 | S2 | SSIM |", "|---|---|---|---|---|---|"]  # fmt: skip
    for arm, d in table.items():
        val = d["val"]
        lines.append(
            f"| {arm} | {d['n']} | {cell(val, 'mae')} | {cell(val, 'mae_S1')} | "
            f"{cell(val, 'mae_S2')} | {val['ssim_mean']:.3f} |"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    global VARIANTS, OUT_NAME, TAU
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--tag", default="", help="suffix of pad_calibration<tag>.json")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = ap.parse_args()
    VARIANTS, OUT_NAME = tuple(args.variants), f"pad_calibration{args.tag}.json"
    TAU = args.tau
    payloads = {}
    for b in args.bundles:
        payloads[b] = evaluate_bundle(b, args.draws, args.seed)
        t = payloads[b]["test"]
        print(
            f"{b.name}: none {t['none']['macro_mae']:.1f}  "
            + "  ".join(
                f"{v}: k5 {t[f'pad_k5/{v}']['macro_mae']:.1f} "
                f"k20 {t[f'pad_k20/{v}']['macro_mae']:.1f} "
                f"oracle {t[f'pad_oracle/{v}']['macro_mae']:.1f}"
                for v in VARIANTS
            ),
            flush=True,
        )
    if args.summary:
        args.summary.mkdir(parents=True, exist_ok=True)
        summarize(payloads, args.summary)


if __name__ == "__main__":
    main()
