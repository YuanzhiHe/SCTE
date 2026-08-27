#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "infer_volume.py" > /dev/null; do sleep 60; done
sleep 20
[ -s runs/BL_i3net.pt ] || $PY scripts/baseline_nets.py train --net i3net --root DATA/aligned_train \
    --steps 20000 --crop 256 --z_shift 2 --amp > runs/BL_i3net.log 2>&1
tail -2 runs/BL_i3net.log
[ -s results/AL_vol_i3net.csv ] || $PY scripts/baseline_nets.py infer --net i3net --root DATA/aligned_test \
    --crop 256 --n 50 --z_shift 2 --amp --csv results/AL_vol_i3net.csv > runs/BL_i3net_infer.log 2>&1
tail -10 runs/BL_i3net_infer.log
echo "== i3net done"
