#!/usr/bin/env bash
# Single sample for the headline nodule-survival numbers; the 6-sample spread map is a
# separate, smaller run. At 512x512 a 6-sample pass costs ~37 min/case - 26 hours for
# the cohort - and the spread map is not what the survival table needs.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=DATA/luna_pairs
C="--root $R --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028"
C="$C --flow_steps 64 --lung_blend 1 --learned_op runs/forward_op_aligned.pt"
C="$C --calibration runs/protocol_aligned.json"
$PY scripts/infer_volume.py $C --samples 1 --save_recon RECON/luna > runs/luna_flow1.log 2>&1
echo "== 单采样完成 $(ls RECON/luna/*_rec.npy | wc -l) 例"
$PY scripts/nodule_survival.py --root $R --recon_dir RECON/luna --label "CTHNet+流匹配" \
    --csv results/luna_survival_full.csv 2>&1 | tail -14
echo "== LUNA16 done"
