#!/usr/bin/env bash
# Screen s_r on 10 volumes, then spend the full 50-volume run on the winner only.
# s_r=0.028 needs no screening run: AL_vol_hybrid.csv already covers all 50, so the
# same 10 cases can just be sliced out of it.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=runs/protocol_aligned.json; OPX=runs/forward_op_aligned.pt
# do not touch the orphaned 0.017 training - wait it out
while pgrep -f "residual_scale 0.017" > /dev/null; do sleep 60; done
for SR in 0.017 0.012; do
  T=hy${SR/./}
  [ -f "runs/$T.pt" ] || $PY -m scte_r.train --data pairs --root DATA/aligned_train --flow \
      --base_recon --residual_scale $SR --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 \
      --epochs 300 --workers 4 --ckpt "runs/$T.pt" > "runs/$T.log" 2>&1
  [ -s "results/AL_scr_$T.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --ckpt "runs/$T.pt" --flow --base_recon --residual_scale $SR --flow_steps 64 \
      --learned_op $OPX --calibration $P --n 10 --csv "results/AL_scr_$T.csv" \
      > "runs/${T}_scr.log" 2>&1
  echo "== s_r=$SR"; grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/${T}_scr.log" | head -5
done
echo "== screening done"
