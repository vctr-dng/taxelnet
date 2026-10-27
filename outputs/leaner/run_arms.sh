#!/usr/bin/env bash
# Leaner taxel head: cnn_v0 width 16 + 2 taxel 3x3 convs with fewer channels
# (model.taxel_channels 32 -> 51k params, 16 -> 35k; the 64-channel reference, 97k, is
# th-h2 in the lab/blacklist worktree, outputs/taxelhead). Grasp-only, glitch blacklist,
# per-pad target scale, 5-fold object CV (fold k, seed k), one model per fold.
set -u
WT=/home/vctr-dng/robotic/taxelnet/.claude/worktrees/leaner
D=/home/vctr-dng/robotic/taxelnet/data/newton/v1
PY=/home/vctr-dng/robotic/taxelnet/.venv/bin/python
OUT=$WT/outputs/leaner/runs
mkdir -p "$OUT/logs"
export PYTHONPATH=$WT/src MLFLOW_DISABLE_AGENT_HINT=1
cd "$WT"

run() {  # taxel_channels fold
  c=$1; f=$2
  name=ln-c$c-f$f
  "$PY" scripts/train.py \
    --data.data_dir "$WT/data_bl/glitch" --data.real_csv Dataset_GraspOnly_Real.csv \
    --data.sim_csv Dataset_GraspOnly_Simulation.csv --data.cache_dir "$WT/data_bl/.cache" \
    --data.test_dir "$D/test" --data.split_strategy shape_fold --data.n_folds 5 --data.fold "$f" \
    --model.name cnn_v0 --model.base_channels 16 --model.taxel_convs 2 --model.taxel_channels "$c" \
    --preprocessing.target_group_key sensor \
    --train.seed "$f" --train.output_dir "$OUT" --train.test_every 50 \
    --tracking.uri http://192.168.193.150:5000 --tracking.experiment taxelnet-taxelhead \
    --tracking.run_name "$name" > "$OUT/logs/$name.log" 2>&1
  echo "$name exit $?"
}
export -f run; export WT D PY OUT

for c in 32 16; do for f in 0 1 2 3 4; do echo "$c $f"; done; done | xargs -P 3 -L 1 bash -c 'run $0 $1'
"$PY" scripts/summarize_runs.py "$OUT" /home/vctr-dng/robotic/taxelnet/.claude/worktrees/blacklist/outputs/taxelhead/runs \
  --out "$WT/outputs/leaner/results.md"
