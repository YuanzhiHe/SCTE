#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
COMMON="--root DATA/public_pairs_test --learned_op runs/forward_op_rplhr.pt --calibration runs/protocol_rplhr_5mm.json --n 50"
[ -s results/vol_realft.csv ] || $PY scripts/infer_volume.py $COMMON --ckpt public_realft.pt \
    --csv results/vol_realft.csv > runs/vol_realft.log 2>&1
echo "== realft done"; tail -7 runs/vol_realft.log
[ -s results/vol_flow.csv ] || $PY scripts/infer_volume.py $COMMON --ckpt public_flow.pt --flow \
    --residual_scale 0.028 --flow_steps 64 --csv results/vol_flow.csv > runs/vol_flow.log 2>&1
echo "== flow done"; tail -7 runs/vol_flow.log
