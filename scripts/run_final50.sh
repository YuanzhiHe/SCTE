#!/usr/bin/env bash
# Final configuration on all 50 test volumes, both outputs of the SAME model:
#   single sample  -> the densitometry report
#   4-sample mean  -> the viewing image
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=$1
[ -s "results/AL_final_s$N.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
    --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028 --flow_steps 64 \
    --lung_blend 1 --samples $N --learned_op runs/forward_op_aligned.pt \
    --calibration runs/protocol_aligned.json --n 50 --csv "results/AL_final_s$N.csv" \
    > "runs/final_s$N.log" 2>&1
echo "== samples=$N done"; grep -E "^  (psnr|lung_psnr|ssim|lung_ssim|laa950|laa910|p15|p10)" "runs/final_s$N.log"
