"""Multi-objective Optuna search of the standard CNN (see taxelnet.hpo).

Usage: uv run python scripts/hpo.py [--hpo.n_trials N] [--train.epochs E] [...]
       uv run python scripts/hpo.py --hpo.report_only true   # rewrite the report only
       uv run python scripts/hpo.py --help                   # every --<section>.<field>

Parallel workers: start the same command in several shells; they share the study
through `--hpo.storage` (sqlite by default). Outputs: <hpo.output_dir>/<hpo.study_name>/
(trial bundles, trials.csv, pareto.csv, pareto.png, study.json). Each trial_<n>/seed<k>/
holds val plots every `--hpo.plot_every` epochs (plots/epoch_<N>/: whole val pair and each
group) and at the best calibrated val MAE (plots/best/, best_cal.pt), the learning curves
and eval_history.csv; with `--hpo.seeds_per_trial` > 1, trial_<n>/plots/ shows violins
over the seeds. The MLflow run of a trial has the params optuna.trial_number / trial_id.
"""

from typing import cast

import polars as pl

from taxelnet.config import HPOConfigs, parse_configs
from taxelnet.hpo import OBJECTIVES, run_study


def main() -> None:
    cfgs = cast(HPOConfigs, parse_configs(cls=HPOConfigs))
    pareto = run_study(cfgs)
    if pareto.is_empty():
        return
    cols = ["trial", *(f"{p}_{k}" for p in ("val", "test") for k in OBJECTIVES)]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
        print(f"Pareto front ({len(pareto)} trials, sorted by val MAE):")
        print(pareto.select(cols))
    print(f"report: {cfgs.hpo.output_dir / cfgs.hpo.study_name}")


if __name__ == "__main__":
    main()
