#!/usr/bin/env bash
# Distortion-perception dial: averaging N samples moves the output toward the
# conditional mean (PSNR up, density tail down). Measures how far PSNR recovers.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for N in 1 4; do
  [ -s "results/AL_smp_$N.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028 --flow_steps 64 \
      --lung_blend 1 --samples $N --learned_op runs/forward_op_aligned.pt \
      --calibration runs/protocol_aligned.json --n 6 --csv "results/AL_smp_$N.csv" \
      > "runs/smp_$N.log" 2>&1
  echo "== samples=$N"; grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/smp_$N.log" | head -5
done
echo "== samples sweep done"
