#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
for T in -600 -400 -300 -200; do
  for IOU in 0 0.3; do
    tag="T${T#-}_iou${IOU/./}"
    [ -s "runs/PR_$tag.log" ] || $PY scripts/hallucination.py --n 8 --thr $T --iou $IOU \
        --recon_dir RECON/s1 --label "流匹配单采样" > "runs/PR_$tag.log" 2>&1
    echo "### thr=$T IoU>=$IOU"; tail -6 "runs/PR_$tag.log" | head -5
  done
done
echo "== PR sweep done"
