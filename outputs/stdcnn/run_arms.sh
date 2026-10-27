#!/usr/bin/env bash
# Standard CNN (cnn_v0: conv blocks + adaptive pooling to the taxel grid, no CoordConv, no
# physics hybrid) on grasp-only data with the glitch blacklist (data_bl/glitch), per-pad
# target scale. Widths 16 and 32.
#   old split: shape split seed 0 (val Cuboid_G), seeds 0-4   -> run names sc-<arm>-s<seed>
#   5-fold:    shape_fold, fold k with seed k                  -> run names sc-<arm>-f<fold>
# <=3 concurrent runs. Calibrated test metric (padgain_k5) is logged by training itself.
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/blacklist
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/stdcnn/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # arm mode(s|f) index
  arm=$1; mode=$2; i=$3
  case $arm in
    cnn16) extra="--model.base_channels 16";;
    cnn32) extra="--model.base_channels 32";;
  esac
  if [ "$mode" = s ]; then
    split="--data.split_strategy shape --data.split_seed 0"
  else
    split="--data.split_strategy shape_fold --data.n_folds 5 --data.fold $i"
  fi
  name=sc-$arm-$mode$i
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_bl/glitch" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_bl/.cache" \
    --data.test_dir "$D/test" $split \
    --model.name cnn_v0 $extra --preprocessing.target_group_key sensor \
    --train.seed "$i" --train.output_dir "$OUT" --train.test_every 50 \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-stdcnn \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

run cnn16 s 0   # first run alone: creates the MLflow experiment
for a in cnn16 cnn32; do for m in s f; do for i in 0 1 2 3 4; do echo "$a $m $i"; done; done; done \
  | grep -v "^cnn16 s 0$" | xargs -P 3 -L 1 bash -c 'run $0 $1 $2'
