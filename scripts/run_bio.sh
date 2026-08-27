#!/usr/bin/env bash
# The published biomarker-multi-task recipe, run on OUR aligned data and scored
# with OUR certificate: does optimising the reported index directly buy accuracy,
# or buy a global HU displacement? Measured, not argued.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=runs/protocol_aligned.json; OPX=runs/forward_op_aligned.pt
while pgrep -f "run_samples.sh" > /dev/null; do sleep 60; done
for W in 1.0 0.3; do
  T=bio${W/./}
  [ -f "runs/$T.pt" ] || $PY -m scte_r.train --data pairs --root DATA/aligned_train --arc \
      --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 --epochs 300 --workers 4 \
      --w_traj 0 --w_bio $W --ckpt "runs/$T.pt" > "runs/$T.log" 2>&1
  [ -s "results/AL_cert_$T.csv" ] || $PY scripts/certify.py --root DATA/aligned_test \
      --ckpt "runs/$T.pt" --calibration $P --learned_op $OPX --lung_mask \
      --csv "results/AL_cert_$T.csv" > "runs/${T}_cert.log" 2>&1
  echo "== w_bio=$W"; grep -E "delta|verdict|LAA-950 MAE|Perc15" "runs/${T}_cert.log" | head -5
  [ -s "results/AL_vol_$T.csv" ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
      --ckpt "runs/$T.pt" --learned_op $OPX --calibration $P --n 50 \
      --csv "results/AL_vol_$T.csv" > "runs/${T}_vol.log" 2>&1
  grep -E "^  (psnr|lung_psnr|laa950|laa910|p15)" "runs/${T}_vol.log" | head -5
done
echo "== biomarker-loss arm done"
