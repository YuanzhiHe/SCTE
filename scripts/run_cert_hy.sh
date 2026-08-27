#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s results/AL_cert_hybrid.csv ] || $PY scripts/certify.py --root DATA/aligned_test \
    --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028 --flow_steps 64 \
    --calibration runs/protocol_aligned.json --learned_op runs/forward_op_aligned.pt \
    --lung_mask --per_lobe --csv results/AL_cert_hybrid.csv > runs/cert_hybrid.log 2>&1
tail -8 runs/cert_hybrid.log
