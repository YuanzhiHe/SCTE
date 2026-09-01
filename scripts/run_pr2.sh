#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
while pgrep -f "run_pr.sh" > /dev/null; do sleep 30; done
for T in -600 -500 -400 -300 -200; do
  for IOU in 0 0.1 0.3 0.5; do
    tag="v2_T${T#-}_iou${IOU/./}"
    [ -s "runs/PR_$tag.log" ] || $PY scripts/hallucination.py --n 8 --thr $T --iou $IOU \
        --recon_dir RECON/s1 --label flow > "runs/PR_$tag.log" 2>&1
  done
done
echo "== PR v2 done"
