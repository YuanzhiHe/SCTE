#!/usr/bin/env bash
# Reconstruct all 50 test volumes so the safety audit stops being an n=8 statement.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="--root DATA/aligned_test --ckpt runs/HY_flow.pt --flow --base_recon --residual_scale 0.028"
C="$C --flow_steps 64 --lung_blend 1 --learned_op runs/forward_op_aligned.pt"
C="$C --calibration runs/protocol_aligned.json --n 50"
[ "$(ls RECON/s1_50/*_rec.npy 2>/dev/null | wc -l)" -ge 50 ] || \
  $PY scripts/infer_volume.py $C --samples 1 --save_recon RECON/s1_50 \
      --csv results/AUD_s1_50.csv > runs/recon50_s1.log 2>&1
echo "== 单采样 $(ls RECON/s1_50/*_rec.npy 2>/dev/null | wc -l) 例"
# 6 samples costs 6x, i.e. ~18 min/case: 15 h for the full 50. The spread map supports
# one secondary claim (localising its own uncertainty), so 20 cases - 2.5x the current
# n=8 at a third of the wall clock - buys more than the extra 30 would.
[ "$(ls RECON/sp_50/*_std.npy 2>/dev/null | wc -l)" -ge 20 ] || \
  $PY scripts/infer_volume.py ${C/--n 50/--n 20} --samples 6 --save_std \
      --save_recon RECON/sp_50 --csv results/AUD_sp_50.csv > runs/recon50_sp.log 2>&1
echo "== 6采样 $(ls RECON/sp_50/*_std.npy 2>/dev/null | wc -l) 例"
echo "== recon50 done"
