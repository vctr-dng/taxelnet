#!/usr/bin/env bash
# Baseline cnn_v0 on the merged indenter + grasp training pair (4 runs, concurrent).
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/grasp-merge
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/merged_baseline
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # name strategy seed
  "$PY" scripts/train.py \
    --data.data_dir "$D/merged" --data.real_csv Dataset_Merged_Real.csv \
    --data.sim_csv Dataset_Merged_Simulation.csv --data.cache_dir "$D/merged/.cache" \
    --data.test_dir "$D/test" --data.split_strategy "$2" --data.split_seed 0 \
    --train.seed "$3" --train.output_dir "$OUT" \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-v0-merged \
    --tracking.run_name "$1" > "$OUT/logs/$1.log" 2>&1
  echo "$1 exit $?"
}

# first run alone so the MLflow experiment and the merged sim cache are created once
run m0-shape-s0 shape 0
run m0-shape-s1 shape 1 &
run m0-shape-s2 shape 2 &
run m0-group-s0 group 0 &
wait
