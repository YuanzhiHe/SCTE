#!/usr/bin/env bash
# Every arm under ONE protocol: whole prepared volume, TotalSegmentator lung mask,
# identical PSNR/SSIM convention. Classical baselines are CPU-only so they can run
# while the GPU is busy.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
N=${N:-50}
for K in nearest linear cubic lanczos; do
  [ -s "results/vol_$K.csv" ] && { echo "== $K done"; continue; }
  echo "== $K"
  $PY scripts/infer_volume.py --root DATA/public_pairs_test --interp $K --n $N \
      --device cpu --csv "results/vol_$K.csv" > "runs/vol_$K.log" 2>&1
  tail -8 "runs/vol_$K.log"
done
echo "== classical baselines done"
