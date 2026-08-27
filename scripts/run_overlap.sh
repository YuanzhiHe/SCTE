#!/usr/bin/env bash
# Does the Hann-blended slab overlap explain why whole-volume flow beats every
# baseline on PSNR while patch-level PSNR DROPS with sampling steps? If the
# overlap is the variance reducer, PSNR should fall as overlap -> 0.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
while [ ! -s results/vol_flow.csv ]; do sleep 60; done
for OV in 0 10 20 30; do
  [ -s "results/ovl_$OV.csv" ] && continue
  $PY scripts/infer_volume.py --root DATA/public_pairs_test --ckpt public_flow.pt --flow \
      --residual_scale 0.028 --flow_steps 64 --learned_op runs/forward_op_rplhr.pt \
      --calibration runs/protocol_rplhr_5mm.json --slab 40 --overlap $OV --n 8 \
      --csv "results/ovl_$OV.csv" > "runs/ovl_$OV.log" 2>&1
  echo "== overlap=$OV"; grep -E "psnr|lung_psnr|ssim|laa950 " "runs/ovl_$OV.log" | head -4
done
echo "== overlap ablation done"
