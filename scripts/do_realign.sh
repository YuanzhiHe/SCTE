#!/usr/bin/env bash
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
for S in public_pairs:aligned_train public_pairs_val:aligned_val; do
  SRC=DATA/${S%%:*}; DST=DATA/${S##*:}
  [ -s "$DST/.done" ] && continue
  $PY scripts/realign_pairs.py --src "$SRC" --dst "$DST" > "runs/realign_${S##*:}.log" 2>&1 && touch "$DST/.done"
  tail -3 "runs/realign_${S##*:}.log"
done
echo REALIGN_DONE
