#!/usr/bin/env bash
# CTHNet backbone (frozen, cached) + flow-matched residual on top.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=runs/protocol_aligned.json; OPX=runs/forward_op_aligned.pt
[ -f runs/HY_flow.pt ] || $PY -m scte_r.train --data pairs --root DATA/aligned_train --flow \
    --base_recon --residual_scale 0.028 --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 \
    --epochs 300 --workers 4 --ckpt runs/HY_flow.pt > runs/HY_flow.log 2>&1
tail -2 runs/HY_flow.log
[ -s results/AL_vol_hybrid.csv ] || $PY scripts/infer_volume.py --root DATA/aligned_test \
    --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028 --flow_steps 64 \
    --learned_op $OPX --calibration $P --n 50 --csv results/AL_vol_hybrid.csv \
    > runs/HY_vol.log 2>&1
tail -12 runs/HY_vol.log
echo "== hybrid done"
