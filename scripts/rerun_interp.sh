#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
for K in nearest linear cubic lanczos; do
  [ -s "results/AL_vol_$K.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --interp $K --n 50 --device cpu --csv "results/AL_vol_$K.csv" > "runs/AL_vol_$K.log" 2>&1
  echo "== $K"; grep -E "^  (psnr|lung_psnr|ssim|laa950)" "runs/AL_vol_$K.log" | head -4
done
echo "== interp done"
