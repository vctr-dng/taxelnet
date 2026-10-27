#!/usr/bin/env bash
# Capacity arms on grasp-only data, global scaling, shape split seed 0, seeds 0-2, <=3 concurrent runs.
# Data: python scripts/build_sensor_data.py --out-dir data_sensor/all
#       (and copy/refresh the sim cache into data_sensor/.cache, or let the first run build it)
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/capacity
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/capacity/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # arm seed
  arm=$1; seed=$2; extra=""
  CF="--model.name cnn_feat --model.input_features z,coord"
  BIG="--model.base_channels 32 --model.convs_per_block 2 --model.norm group"
  case $arm in
    cnn_v0)    extra="";;
    w32)       extra="--model.base_channels 32";;
    w64)       extra="--model.base_channels 64";;
    feat)      extra="$CF";;
    featbig)   extra="$CF $BIG";;
    fp)        extra="--model.name footprint --train.lr 1e-2";;
    fpcnn)     extra="--model.name footprint_cnn --train.residual_l1_weight 0.05 --train.physics_lr 1e-2";;
    fpcnncoord) extra="--model.name footprint_cnn --model.residual_backbone cnn_feat --model.input_features z,coord --train.residual_l1_weight 0.05 --train.physics_lr 1e-2";;
  esac
  name=cap-$arm-s$seed
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_sensor/all" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_sensor/.cache" \
    --data.test_dir "$D/test" --data.split_strategy shape --data.split_seed 0 \
    --train.seed "$seed" --train.output_dir "$OUT" --train.test_every 50 $extra \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-capacity \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

run cnn_v0 0   # first run alone: creates the MLflow experiment
for a in cnn_v0 w32 w64 feat featbig fp fpcnn fpcnncoord; do for s in 0 1 2; do echo "$a $s"; done; done \
  | grep -v "^cnn_v0 0$" | xargs -P 3 -L 1 bash -c 'run $0 $1'
