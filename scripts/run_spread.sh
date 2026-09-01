#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "run_pr.sh" > /dev/null; do sleep 30; done
$PY scripts/infer_volume.py --root DATA/aligned_test --ckpt runs/HY_flow.pt --flow \
    --base_recon --residual_scale 0.028 --flow_steps 64 --lung_blend 1 \
    --learned_op runs/forward_op_aligned.pt --calibration runs/protocol_aligned.json \
    --n 8 --samples 6 --save_std --save_recon RECON/spread --csv results/HAL_spread.csv \
    > runs/spread.log 2>&1
echo "== 6 采样 + 离散度图完成"; tail -6 runs/spread.log
