#!/usr/bin/env bash
# Transfer between indenter and grasp data, standard CNN (cnn_v0 base 16), <=3 concurrent runs.
#   ind-pre-s<k>      indenter only (shape split seed 0), single target scale, seeds 0-4
#   ind2grasp-s<k>    init from ind-pre-s<k>, fine-tuned on grasp (glitch blacklist, per-pad
#                     scale, shape split seed 0); ind2grasp-f<k>: same on grasp fold k
#   grasp2ind-s<k>    init from sc-cnn16-s<k> (grasp only, outputs/stdcnn), fine-tuned on
#                     indenter (shape split seed 0, single target scale)
# Fine-tuning: lr 3e-4, 150 epochs (pre-training: lr 1e-3, 300 epochs). The input scaler of
# the pre-trained bundle is reused (train.init_from), the target scaler is refitted.
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/blacklist
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/transfer/runs
SC=$WT/outputs/stdcnn/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

# the standard CNN launcher must be done (grasp2ind starts from its bundles)
until [ "$(grep -c exit "$WT/outputs/stdcnn/launcher.log")" -ge 20 ]; do sleep 30; done

COMMON="--model.name cnn_v0 --model.base_channels 16 --data.test_dir $D/test --train.test_every 50
  --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-transfer"
IND="--data.data_dir $D/training --data.real_csv Dataset_Indenter_Real.csv
  --data.sim_csv Dataset_Indenter_Simulation.csv --data.cache_dir $WT/data_bl/.cache
  --data.split_strategy shape --data.split_seed 0 --preprocessing.target_group_key none"
GRASP="--data.data_dir $WT/data_bl/glitch --data.real_csv Dataset_GraspOnly_Real.csv
  --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir $WT/data_bl/.cache
  --preprocessing.target_group_key sensor"
FT="--train.lr 3e-4 --train.epochs 150"

bundle() { ls -d "$1"/*_"$2" | tail -1; }

run() {  # kind index
  kind=$1; i=$2
  case $kind in
    ind-pre)  name=ind-pre-s$i; args="$IND --train.seed $i";;
    ind2grasp-s) name=ind2grasp-s$i
      args="$GRASP --data.split_strategy shape --data.split_seed 0 $FT --train.seed $i
        --train.init_from $(bundle "$OUT" ind-pre-s$i)";;
    ind2grasp-f) name=ind2grasp-f$i
      args="$GRASP --data.split_strategy shape_fold --data.n_folds 5 --data.fold $i $FT
        --train.seed $i --train.init_from $(bundle "$OUT" ind-pre-s$i)";;
    grasp2ind) name=grasp2ind-s$i
      args="$IND $FT --train.seed $i --train.init_from $(bundle "$SC" sc-cnn16-s$i)";;
  esac
  "$PY" scripts/train.py $COMMON $args --train.output_dir "$OUT" \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run bundle; export WT D PY OUT SC COMMON IND GRASP FT

run ind-pre 0   # first run alone: creates the MLflow experiment and the indenter cache
printf '%s\n' "ind-pre 1" "ind-pre 2" "ind-pre 3" "ind-pre 4" \
  "grasp2ind 0" "grasp2ind 1" "grasp2ind 2" "grasp2ind 3" "grasp2ind 4" \
  | xargs -P 3 -L 1 bash -c 'run $0 $1'
for k in ind2grasp-s ind2grasp-f; do for i in 0 1 2 3 4; do echo "$k $i"; done; done \
  | xargs -P 3 -L 1 bash -c 'run $0 $1'
