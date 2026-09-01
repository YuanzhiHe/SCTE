#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--root DATA/aligned_test --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028"
C="$C --learned_op runs/forward_op_aligned.pt --calibration runs/protocol_aligned.json --n 10 --lung_blend 1"
run () { # tag, extra args
  [ -s "results/GG_$1.csv" ] && { echo "== $1 已有"; return; }
  s=$(date +%s)
  $PY scripts/infer_volume.py $C $2 --csv "results/GG_$1.csv" > "runs/GG_$1.log" 2>&1
  echo "== $1  用时 $(( ($(date +%s)-s)/60 )) 分钟"
  grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/GG_$1.log" | head -5
}
run base   "--flow_steps 64"
run gate80 "--flow_steps 64 --grad_gate 80"
run adapt  "--flow_steps 64 --grad_gate 80 --grad_steps 8,64"
echo "== gradient gating sweep done"
