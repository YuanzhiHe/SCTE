#!/usr/bin/env bash
# Zero-shot on LUNA16: models trained on RPLHR, applied to a different cohort with a
# SIMULATED thick series. Two domain gaps at once (scanner and degradation), so this
# is a generalisation test, not a like-for-like comparison.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=DATA/luna_pairs
# 1) backbone reconstructions (CTHNet, trained on the aligned RPLHR split)
if ! ls $R/*_base.npy >/dev/null 2>&1; then
  echo "== CTHNet 零样本"
  $PY scripts/baseline_nets.py infer --net cthnet --root $R --ckpt runs/BL_cthnet.pt \
      --crop 256 --z_shift 2 --amp --grad_ckpt --save_recon $R > runs/luna_cthnet.log 2>&1
  echo "   $(ls $R/*_base.npy 2>/dev/null | wc -l) 例完成"
fi
# 2) flow residual on the frozen backbone, with the spread map
if ! ls RECON/luna/*_rec.npy >/dev/null 2>&1; then
  echo "== 流匹配 零样本 (6 采样 + 离散度)"
  $PY scripts/infer_volume.py --root $R --ckpt runs/HY_flow.pt --flow --base_recon \
      --residual_scale 0.028 --flow_steps 64 --lung_blend 1 --samples 6 --save_std \
      --learned_op runs/forward_op_aligned.pt --calibration runs/protocol_aligned.json \
      --save_recon RECON/luna > runs/luna_flow.log 2>&1
  echo "   $(ls RECON/luna/*_rec.npy 2>/dev/null | wc -l) 例完成"
fi
echo "== 逐结节存活"
$PY scripts/nodule_survival.py --root $R --recon_dir RECON/luna --label "CTHNet+流匹配" \
    --csv results/luna_survival_full.csv 2>&1 | tail -14
echo "== LUNA16 done"
