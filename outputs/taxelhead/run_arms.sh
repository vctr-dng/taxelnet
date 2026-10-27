#!/usr/bin/env bash
# 3x3 convs on the 7x4 taxel map after the pooling (model.taxel_convs 0 | 1 | 2), cnn_v0 width
# 16, grasp-only data with the glitch blacklist, per-pad target scale, 5-fold object CV
# (fold k, seed k), one model per fold (no ensembling). <=3 concurrent runs.
# Test metrics: uncalibrated, test-pad y_scale (testscale_*), calibrated gain (padgain_*).
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/blacklist
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/taxelhead/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # n_taxel_convs fold
  h=$1; f=$2
  name=th-h$h-f$f
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_bl/glitch" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_bl/.cache" \
    --data.test_dir "$D/test" --data.split_strategy shape_fold --data.n_folds 5 --data.fold "$f" \
    --model.name cnn_v0 --model.base_channels 16 --model.taxel_convs "$h" \
    --preprocessing.target_group_key sensor \
    --train.seed "$f" --train.output_dir "$OUT" --train.test_every 50 \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-taxelhead \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

run 0 0   # first run alone: creates the MLflow experiment
for h in 0 1 2; do for f in 0 1 2 3 4; do echo "$h $f"; done; done \
  | grep -v "^0 0$" | xargs -P 3 -L 1 bash -c 'run $0 $1'
"$PY" scripts/summarize_runs.py "$OUT" --out "$WT/outputs/taxelhead/results.md"
