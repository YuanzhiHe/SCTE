#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "run_paren.sh" > /dev/null; do sleep 60; done
C="--root DATA/aligned_test --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028"
C="$C --learned_op runs/forward_op_aligned.pt --calibration runs/protocol_aligned.json --n 8 --lung_blend 1 --flow_steps 64"
[ -d RECON/s1 ] || $PY scripts/infer_volume.py $C --samples 1 --save_recon RECON/s1 \
    --csv results/HAL_s1.csv > runs/HAL_s1.log 2>&1
echo "== 单采样重建已存盘"
[ -d RECON/s4 ] || $PY scripts/infer_volume.py $C --samples 4 --save_recon RECON/s4 \
    --csv results/HAL_s4.csv > runs/HAL_s4.log 2>&1
echo "== 4采样重建已存盘"
$PY scripts/hallucination.py --n 8 --recon_dir RECON/s1 --label "流匹配单采样" 2>&1 | tail -8
$PY scripts/hallucination.py --n 8 --recon_dir RECON/s4 --label "流匹配4采样" 2>&1 | tail -6
echo "== hallucination audit done"
