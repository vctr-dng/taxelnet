#!/usr/bin/env bash
# Grasp-only cnn_v0, shape split seed 0, arms A-D x seeds 0-2, at most 3 concurrent runs.
# Data: python scripts/build_sensor_data.py --out-dir data_sensor/all
#       python scripts/build_sensor_data.py --out-dir data_sensor/noglitch --drop-above 10000
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/sensornorm
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/sensornorm/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # arm seed
  arm=$1; seed=$2; data=all; extra=""
  case $arm in
    A) extra="";;
    B) extra="--preprocessing.target_group_key sensor";;
    C) extra="--preprocessing.target_group_key sensor --preprocessing.target_scaler per_taxel_zscore";;
    D) data=noglitch; extra="--preprocessing.target_group_key sensor";;
  esac
  name=sn-$arm-s$seed
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_sensor/$data" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_sensor/.cache" \
    --data.test_dir "$D/test" --data.split_strategy shape --data.split_seed 0 \
    --train.seed "$seed" --train.output_dir "$OUT" $extra \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-sensornorm \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

# first run alone: creates the MLflow experiment and the sim caches
run A 0
run D 0
printf '%s\n' "A 1" "A 2" "B 0" "B 1" "B 2" "C 0" "C 1" "C 2" "D 1" "D 2" | xargs -P 3 -L 1 bash -c 'run $0 $1'
