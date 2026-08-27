#!/usr/bin/env bash
# How many real paired cases does the flow decoder actually need?
# Matched GRADIENT-STEP budget across arms (epochs scaled inversely with N), so the
# only thing that varies is how many distinct cases the model has seen.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
run () {  # $1=tag $2=root $3=epochs
  [ -f "runs/SC_$1.pt" ] && { echo "== $1 already trained"; return; }
  echo "== training $1 (root=$2, epochs=$3)"
  $PY -m scte_r.train --data pairs --root "$2" --flow --residual_scale 0.028 \
      --calibration runs/protocol_rplhr_5mm.json --clip 1.0 --ema 0.999 --lr 1e-3 \
      --epochs "$3" --workers 4 --ckpt "runs/SC_$1.pt" > "runs/SC_$1.log" 2>&1
  tail -2 "runs/SC_$1.log"
}
run n20 DATA/scale_n20 1275
run n45 DATA/scale_n45  567
run n85 DATA/public_pairs 300
for N in n20 n45 n85; do
  [ -f "results/scale_$N.csv" ] && continue
  echo "== certifying $N"
  $PY scripts/certify.py --root DATA/public_pairs_test --ckpt "runs/SC_$N.pt" --flow \
      --residual_scale 0.028 --flow_steps 64 --calibration runs/protocol_rplhr_5mm.json \
      --learned_op runs/forward_op_rplhr.pt --lung_mask \
      --csv "results/scale_$N.csv" > "runs/SC_${N}_cert.log" 2>&1
  tail -6 "runs/SC_${N}_cert.log"
done
echo "== scaling sweep done"
