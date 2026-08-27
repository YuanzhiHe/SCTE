#!/usr/bin/env bash
# The residual normalisation was off by 4.2x: the flow target u = (x-up(y))/s_r had
# std 4.19, not 1. The published sweep {0.020..0.050} never reached the regime where
# the design's own rationale (prior and target at the same scale) actually holds.
# Measured lung residual std = 117 HU -> s_r = 0.117 kHU is the matched value.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
while pgrep -f "baseline_nets.py" > /dev/null; do sleep 60; done
sleep 30
for SR in 0.117 0.080 0.160; do
  T=sr${SR/./}
  [ -f "runs/SR_$T.pt" ] || $PY -m scte_r.train --data pairs --root DATA/public_pairs --flow \
      --residual_scale $SR --calibration runs/protocol_rplhr_5mm.json --clip 1.0 --ema 0.999 \
      --lr 1e-3 --epochs 300 --workers 4 --ckpt "runs/SR_$T.pt" > "runs/SR_$T.log" 2>&1
  [ -s "results/sr_$T.csv" ] || $PY scripts/certify.py --root DATA/public_pairs_test \
      --ckpt "runs/SR_$T.pt" --flow --residual_scale $SR --flow_steps 64 \
      --calibration runs/protocol_rplhr_5mm.json --learned_op runs/forward_op_rplhr.pt \
      --lung_mask --csv "results/sr_$T.csv" > "runs/SR_${T}_cert.log" 2>&1
  echo "== s_r=$SR"; tail -6 "runs/SR_${T}_cert.log"
done
echo "== residual-scale fix done"
