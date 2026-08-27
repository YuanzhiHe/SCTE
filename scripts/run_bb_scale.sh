#!/usr/bin/env bash
# Does the 10.6M-parameter backbone saturate by 85 cases, or is it still climbing?
# The flow decoder's scaling curve says nothing about a transformer this size, and
# whether 203 real pairs suffice depends on THIS curve, not that one.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s runs/BB_n45.pt ] || $PY scripts/baseline_nets.py train --net cthnet --root DATA/aligned_n45 \
    --steps 20000 --crop 256 --z_shift 2 --amp --grad_ckpt --ckpt runs/BB_n45.pt \
    > runs/BB_n45.log 2>&1
tail -2 runs/BB_n45.log
[ -s results/AL_vol_bb45.csv ] || $PY scripts/baseline_nets.py infer --net cthnet \
    --root DATA/aligned_test --ckpt runs/BB_n45.pt --crop 256 --n 50 --z_shift 2 \
    --amp --grad_ckpt --csv results/AL_vol_bb45.csv > runs/BB_n45_infer.log 2>&1
tail -9 runs/BB_n45_infer.log
echo "== backbone scaling done"
