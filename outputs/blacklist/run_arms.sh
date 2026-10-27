#!/usr/bin/env bash
# Blacklist arms on grasp-only data: glitch (26 rows dropped) vs all160 (every flagged row dropped).
# Model footprint_cnn + CoordConv residual, per-pad target scale, shape split seed 0 (val = Cuboid_G,
# identical in both arms), seeds 0-4, <=3 concurrent runs. Then per-test-pad calibration.
# Data: python outputs/blacklist/make_lists.py
#       python scripts/build_sensor_data.py --out-dir data_bl/glitch --exclude-names outputs/blacklist/glitch_names.txt
#       python scripts/build_sensor_data.py --out-dir data_bl/all160 --exclude-names outputs/blacklist/all160_names.txt
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/blacklist
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/blacklist/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # arm seed
  arm=$1; seed=$2
  name=bl-$arm-s$seed
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_bl/$arm" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_bl/.cache" \
    --data.test_dir "$D/test" --data.split_strategy shape --data.split_seed 0 \
    --model.name footprint_cnn --model.residual_backbone cnn_feat --model.input_features z,coord \
    --train.residual_l1_weight 0.05 --train.physics_lr 1e-2 \
    --preprocessing.target_group_key sensor \
    --train.seed "$seed" --train.output_dir "$OUT" --train.test_every 50 \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-blacklist \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

run glitch 0   # first run alone: creates the MLflow experiment
for a in glitch all160; do for s in 0 1 2 3 4; do echo "$a $s"; done; done \
  | grep -v "^glitch 0$" | xargs -P 3 -L 1 bash -c 'run $0 $1'

"$PY" scripts/pad_calibrate_eval.py "$OUT"/*_bl-* --summary "$WT/outputs/blacklist"
