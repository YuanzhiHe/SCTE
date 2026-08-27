#!/usr/bin/env bash
# Two-stage: screen s_r on 10 volumes (35 min each), then spend the full 50-volume
# run on the winner only. A full sweep at n=50 would be 6 h for one tuning decision.
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
  [ -s "results/AL_scr_$T.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --ckpt "runs/$T.pt" --flow --base_recon --residual_scale $SR --flow_steps 64 \
      --learned_op $OPX --calibration $P --n 10 --csv "results/AL_scr_$T.csv" \
      > "runs/${T}_scr.log" 2>&1
  echo "== s_r=$SR (10 例筛选)"; grep -E "^  (psnr|lung_psnr|ssim|laa950|laa910)" "runs/${T}_scr.log" | head -5
done
# 同一 10 例上给 s_r=0.028 的旧混合臂也算一遍，作为筛选基准
[ -s results/AL_scr_hy0028.csv ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
    --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028 --flow_steps 64 \
    --learned_op $OPX --calibration $P --n 10 --csv results/AL_scr_hy0028.csv \
    > runs/hy0028_scr.log 2>&1
echo "== screening done"
