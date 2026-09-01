#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--root DATA/aligned_test --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028"
C="$C --learned_op runs/forward_op_aligned.pt --calibration runs/protocol_aligned.json --n 10 --lung_blend 1 --flow_steps 64"
for T in -600 -700; do
  N=paren${T#-}
  [ -s "results/GG_$N.csv" ] || $PY scripts/infer_volume.py $C --paren_gate $T \
      --csv "results/GG_$N.csv" > "runs/GG_$N.log" 2>&1
  echo "== 实质门控 < $T HU"; grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/GG_$N.log" | head -5
done
echo "== parenchyma gating done"
