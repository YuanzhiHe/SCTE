#!/usr/bin/env bash
# Published baselines, trained on OUR split and scored under OUR protocol. Serial.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEPS=${STEPS:-20000}
# CTHNet needs bf16 + the upstream blocks' own gradient checkpointing to fit at the
# 256 crop its patch embedding is baked to; neither changes the network's maths.
declare -A EXTRA=( [tvsrn]="" [cthnet]="--amp --grad_ckpt" )
for NET in tvsrn cthnet; do
  if [ ! -s "runs/BL_$NET.pt" ]; then
    echo "== training $NET ($STEPS steps, crop 256)"
    $PY scripts/baseline_nets.py train --net $NET --root DATA/public_pairs \
        --steps $STEPS --crop 256 ${EXTRA[$NET]} > "runs/BL_$NET.log" 2>&1 \
        || { echo "!! $NET train failed"; tail -3 "runs/BL_$NET.log"; continue; }
    tail -2 "runs/BL_$NET.log"
  fi
  if [ ! -s "results/vol_$NET.csv" ]; then
    echo "== inferring $NET"
    $PY scripts/baseline_nets.py infer --net $NET --root DATA/public_pairs_test \
        --crop 256 --n 50 ${EXTRA[$NET]} --csv "results/vol_$NET.csv" \
        > "runs/BL_${NET}_infer.log" 2>&1 \
        || { echo "!! $NET infer failed"; tail -3 "runs/BL_${NET}_infer.log"; continue; }
    tail -10 "runs/BL_${NET}_infer.log"
  fi
done
echo "== published baselines done"
