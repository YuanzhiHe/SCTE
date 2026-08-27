#!/usr/bin/env bash
# s_r for the hybrid: CTHNet leaves a 53.9 HU residual, not the 89.5 HU that plain
# up-sampling leaves, so the injection scale tuned for one is wrong for the other.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=runs/protocol_aligned.json; OPX=runs/forward_op_aligned.pt
for SR in 0.017 0.012 0.022; do
  T=hy${SR/./}
  [ -f "runs/$T.pt" ] || $PY -m scte_r.train --data pairs --root DATA/aligned_train --flow \
      --base_recon --residual_scale $SR --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 \
      --epochs 300 --workers 4 --ckpt "runs/$T.pt" > "runs/$T.log" 2>&1
  [ -s "results/AL_vol_$T.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --ckpt "runs/$T.pt" --flow --base_recon --residual_scale $SR --flow_steps 64 \
      --learned_op $OPX --calibration $P --n 50 --csv "results/AL_vol_$T.csv" \
      > "runs/${T}_vol.log" 2>&1
  echo "== s_r=$SR"; grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/${T}_vol.log" | head -5
done
echo "== hybrid sweep done"
