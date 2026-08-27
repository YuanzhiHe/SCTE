#!/usr/bin/env bash
# Everything re-run on the z-aligned pairs. Order matters: the operator and the
# certificate thresholds are inputs to every arm below them.
set -uo pipefail
PY=/home/prinlab/miniconda3/envs/scte/bin/python
cd /home/prinlab/SCTE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TR=DATA/aligned_train; TE=DATA/aligned_test
P=runs/protocol_aligned.json; OPX=runs/forward_op_aligned.pt
say () { echo "===== $* ($(date +%H:%M)) "; }

say "1 protocol constants"
[ -s $P ] || $PY scripts/fit_protocol_operator.py --root $TR --patch 40 64 64 \
    --protocol rplhr/aligned/r5 --out $P > runs/AL_protocol.log 2>&1
tail -3 runs/AL_protocol.log

say "2 learned forward operator"
[ -s $OPX ] || $PY scripts/fit_forward_operator.py --root $TR --out $OPX \
    --epochs 400 > runs/AL_fwdop.log 2>&1
tail -3 runs/AL_fwdop.log

say "3 flow decoder"
[ -f runs/AL_flow.pt ] || $PY -m scte_r.train --data pairs --root $TR --flow \
    --residual_scale 0.028 --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 \
    --epochs 300 --workers 4 --ckpt runs/AL_flow.pt > runs/AL_flow.log 2>&1
tail -2 runs/AL_flow.log

say "4 supervised regression (ours)"
[ -f runs/AL_reg.pt ] || $PY -m scte_r.train --data pairs --root $TR --arc \
    --calibration $P --clip 1.0 --ema 0.999 --lr 1e-3 --epochs 300 --workers 4 \
    --ckpt runs/AL_reg.pt > runs/AL_reg.log 2>&1
tail -2 runs/AL_reg.log

say "5 certificate null on TRAIN -> thresholds"
[ -s results/AL_oracle.csv ] || {
  $PY scripts/certify.py --root $TR --oracle --calibration $P --learned_op $OPX \
      --lung_mask --csv results/AL_oracle.csv > runs/AL_oracle.log 2>&1
  $PY - results/AL_oracle.csv $P <<'PYEOF' >> runs/AL_oracle.log 2>&1
import csv, json, sys, numpy as np
rows=list(csv.DictReader(open(sys.argv[1]))); g=lambda k: np.array([float(r[k]) for r in rows])
d,rs=g('delta_HU'),g('rho_struct'); s=np.array([float(r['s']) for r in rows if r['s']!=''])
tau=lambda a: float(abs(np.mean(a))+2*np.std(a))
T=dict(tau_delta=round(tau(d),2),tau_rho=round(tau(rs),3),tau_s=round(tau(s),2) if len(s) else 2.0)
print('null:',T); p=json.load(open(sys.argv[2])); p.update(T); json.dump(p,open(sys.argv[2],'w'),indent=1)
PYEOF
}
tail -2 runs/AL_oracle.log

say "6 patch-level certificates"
for A in flow reg; do
  EX=""; [ $A = flow ] && EX="--flow --residual_scale 0.028 --flow_steps 64"
  [ -s results/AL_cert_$A.csv ] || $PY scripts/certify.py --root $TE --ckpt runs/AL_$A.pt $EX \
      --calibration $P --learned_op $OPX --lung_mask --per_lobe \
      --csv results/AL_cert_$A.csv > runs/AL_cert_$A.log 2>&1
  echo "-- $A"; tail -5 runs/AL_cert_$A.log
done

say "7 published baselines (trained on the aligned split)"
declare -A EXTRA=( [tvsrn]="" [cthnet]="--amp --grad_ckpt" )
for NET in tvsrn cthnet; do
  [ -s runs/BL_$NET.pt ] || $PY scripts/baseline_nets.py train --net $NET --root $TR \
      --steps 20000 --crop 256 --z_shift 2 ${EXTRA[$NET]} > runs/BL_$NET.log 2>&1
  tail -1 runs/BL_$NET.log
  [ -s results/AL_vol_$NET.csv ] || $PY scripts/baseline_nets.py infer --net $NET --root $TE \
      --crop 256 --n 50 --z_shift 2 ${EXTRA[$NET]} --csv results/AL_vol_$NET.csv \
      > runs/BL_${NET}_infer.log 2>&1
  echo "-- $NET"; tail -9 runs/BL_${NET}_infer.log
done

say "8 whole-volume: ours"
[ -s results/AL_vol_reg.csv ] || $PY scripts/infer_volume.py --root $TE --ckpt runs/AL_reg.pt \
    --learned_op $OPX --calibration $P --n 50 --csv results/AL_vol_reg.csv > runs/AL_vol_reg.log 2>&1
[ -s results/AL_vol_flow.csv ] || $PY scripts/infer_volume.py --root $TE --ckpt runs/AL_flow.pt --flow \
    --residual_scale 0.028 --flow_steps 64 --learned_op $OPX --calibration $P --n 50 \
    --csv results/AL_vol_flow.csv > runs/AL_vol_flow.log 2>&1
echo "== ALL DONE"
