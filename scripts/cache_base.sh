#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for S in aligned_train aligned_val aligned_test; do
  [ -n "$(ls DATA/$S/*_base.npy 2>/dev/null | head -1)" ] && { echo "== $S cached"; continue; }
  echo "== caching CTHNet reconstructions for $S"
  $PY scripts/baseline_nets.py infer --net cthnet --root DATA/$S --crop 256 \
      --z_shift 2 --amp --grad_ckpt --save_recon DATA/$S \
      --csv results/AL_base_$S.csv > runs/cache_$S.log 2>&1 || { echo "!! $S failed"; tail -3 runs/cache_$S.log; }
  echo "  $(ls DATA/$S/*_base.npy 2>/dev/null | wc -l) volumes cached"
done
echo "== base cache done"
