#!/usr/bin/env bash
# ARC-CT on a private paired cohort — one command, every stage resumable.
# Designed for LIMITED, SUPERVISED ACCESS: nothing downloads, nothing phones home,
# and no image data ever leaves the machine (only derived per-case numbers).
#
#   bash scripts/run_private.sh dryrun /data/hebei hebei                # plumbing test
#   bash scripts/run_private.sh full   /data/hebei hebei /data/henan    # THE study design
#   bash scripts/run_private.sh full   /data/hebei hebei                # single cohort, 50/50
#
# $2 = a directory with ONE SUB-DIRECTORY PER CASE, each holding the 1 mm and the
#      5 mm series (DICOM folders or NIfTI files). $3 = a short cohort tag.
# $4 = OPTIONAL external-validation cases dir. When given, $2 is the internal
#      TRAINING cohort and $4 is the held-out EXTERNAL cohort — nothing is ever
#      fitted on $4. This is the design in the project PPT (Hebei 203 internal /
#      Henan 491 external). Without $4 the single cohort is split 50/50 instead.
#
# Re-running skips every stage whose output already exists; FORCE=1 re-does all.
set -euo pipefail
MODE=${1:?usage: run_private.sh dryrun|full <cases_dir> <cohort_tag>}
CASES=${2:?}
TAG=${3:?}
EXT=${4:-}
PY=${PY:-python3}
FORCE=${FORCE:-0}
# Set RESCALE="3072 -1024" when the site ships normalised values instead of HU.
RESCALE=${RESCALE:-}
RESCALE_ARG=""; [ -n "$RESCALE" ] && RESCALE_ARG="--rescale_hu $RESCALE"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/PRIVATE/$TAG
PAIRS=$OUT/pairs
PAIRS_EXT=$OUT/pairs_external
TRAIN=$PAIRS/train
TEST=$PAIRS/test
PROTO=$OUT/protocol_$TAG.json
FWDOP=$OUT/forward_op_$TAG.pt
RES=$OUT/results
STEPS=64                     # sampling steps: 64 is the knee (report §9.3)
SR=0.028                     # residual scale public_flow.pt was trained at
LIMIT=""; EPOCHS=300; NVOL=0; BSTEPS=20000
MASK_CSV=""
# 6 (not 3) because stage 3 splits in half and stage 4 needs >= 3 cases to fit on.
[ "$MODE" = dryrun ] && { LIMIT="--limit 6"; EPOCHS=3; STEPS=16; NVOL=2; BSTEPS=30; }
mkdir -p "$PAIRS" "$RES"
[ -n "$EXT" ] && mkdir -p "$PAIRS_EXT"
cd "$ROOT"
# skip <marker-file> -> returns 0 (=skip) when the stage is already done
skip() { [ "$FORCE" = 0 ] && [ -s "$1" ]; }
CALROOT=""   # where the scanner calibration is measured; set by stage 3
echo "== cohort=$TAG mode=$MODE cases=$CASES -> $OUT"

echo "== [1/11] pair + align + anonymise (writes ID_MAP_DO_NOT_EXPORT.csv)"
if skip "$RES/01_pairing.log"; then echo "   (already done)"; else
$PY scripts/prep_pairs.py --cases "$CASES" --out "$PAIRS" --downsample 5 \
    --thin_hint 1mm --thick_hint 5mm --anonymise $RESCALE_ARG $LIMIT 2>&1 | tee "$RES/01_pairing.log"
if [ -n "$EXT" ]; then
  echo "-- external validation cohort: $EXT"
  $PY scripts/prep_pairs.py --cases "$EXT" --out "$PAIRS_EXT" --downsample 5 \
      --thin_hint 1mm --thick_hint 5mm --anonymise $RESCALE_ARG $LIMIT 2>&1 | tee -a "$RES/01_pairing.log"
fi
fi

echo "== [2/11] anatomical lung masks (TotalSegmentator) — REQUIRED for clinical LAA"
# Without these the LAA denominator is an HU window, which is NOT clinical LAA-950
# (report §8.1: 9.9-13.6 % vs 0.34-1.57 % on the same scan). If TS is unavailable,
# the run continues but every LAA number below must be labelled non-clinical.
if skip "$RES/02_masks.log"; then echo "   (already done)"; else
$PY scripts/make_lung_masks.py --root "$PAIRS" 2>&1 | tee "$RES/02_masks.log" || \
  echo "!! TotalSegmentator failed — LAA below is HU-window, NOT clinical" | tee -a "$RES/02_masks.log"
[ -n "$EXT" ] && { $PY scripts/make_lung_masks.py --root "$PAIRS_EXT" 2>&1 | tee -a "$RES/02_masks.log" || \
  echo "!! TotalSegmentator failed on the external cohort" | tee -a "$RES/02_masks.log"; }
fi
echo "-- alignment self-check (5 min): are the pairs actually on the operator grid?"
# The single most expensive mistake this pipeline can make is a sub-slab z offset:
# it is invisible in every log, it handicaps the classical baselines by ~3 dB, and it
# silently shifts the operator the certificate is built on. Catch it HERE, not in
# stage 11. Costs nothing and writes nothing.
$PY scripts/realign_pairs.py --src "$PAIRS" --check --limit 20 2>&1 | tail -4 | tee "$RES/01_align_check.log"
if grep -q "ALIGNMENT CHECK FAILED" "$RES/01_align_check.log"; then
  echo "!! stage 1 did not align the pairs - stopping. See $RES/01_align_check.log"; exit 1
fi
if [ -n "$EXT" ]; then
  # The external cohort carries the PRIMARY result, so its check must gate too. An
  # earlier version only logged it, and a real run continued past a genuine sub-slab
  # offset on the external cohort - the one cohort whose numbers are the contribution.
  $PY scripts/realign_pairs.py --src "$PAIRS_EXT" --check --limit 20 2>&1 \
      | tail -4 | tee "$RES/01_align_check_ext.log"
  if grep -q "ALIGNMENT CHECK FAILED" "$RES/01_align_check_ext.log"; then
    echo
    echo "!! 外部队列未落在算子网格上。主结果全部出自这个队列，不能带着偏移跑。"
    echo "   修法（精确重切，不插值，不会引入新误差；掩膜会一并重切）："
    echo "     $PY scripts/realign_pairs.py --src $PAIRS_EXT --dst ${PAIRS_EXT}_aligned"
    echo "     mv $PAIRS_EXT ${PAIRS_EXT}_misaligned && mv ${PAIRS_EXT}_aligned $PAIRS_EXT"
    echo "     rm -f $RES/03_split.txt   # 然后重跑同一条命令，阶段 1-2 会自动跳过"
    echo
    echo "   确实要带着偏移继续（不推荐，会污染主结果）：ALLOW_MISALIGNED=1"
    [ "${ALLOW_MISALIGNED:-0}" = 1 ] || exit 1
    echo "   !! ALLOW_MISALIGNED=1 —— 外部队列结果已被污染，报告时必须声明"
  fi
fi

MASK=""; ls "$PAIRS"/*_lung.npy >/dev/null 2>&1 && MASK="--lung_mask"
[ -z "$MASK" ] && echo "!! no lung masks found — proceeding WITHOUT --lung_mask"

echo "== [3/11] deterministic TRAIN/TEST split"
# NOSPLIT=1 for a cohort too small to halve (< 6 cases): everything is fitted and
# measured on the same scans. The zero-shot arms stay valid (no training touches
# them); the fine-tuned arms become IN-SAMPLE and must be labelled as such.
if [ -n "$EXT" ]; then
  N_TR=$(ls "$PAIRS"/*_thin.npy 2>/dev/null | wc -l); N_TE=$(ls "$PAIRS_EXT"/*_thin.npy 2>/dev/null | wc -l)
  TRAIN=$PAIRS; TEST=$PAIRS_EXT
  if [ "${EXT_CAL:-0}" -gt 0 ]; then
    # The protocol constants and the certificate null are PHYSICAL measurements of a
    # scanner, not learned parameters - but fitting them needs 1 mm ground truth, so
    # measuring them on the external cohort would spend its ground truth. Carve out a
    # calibration subset instead: model weights stay purely external, the scanner
    # calibration is measured on held-out cases, and the reported cases stay untouched.
    CAL=$PAIRS_EXT/cal; TEST=$PAIRS_EXT/test
    mkdir -p "$CAL" "$TEST"
    $PY - "$PAIRS_EXT" "$CAL" "$TEST" "${EXT_CAL}" <<'PYEOF' | tee "$RES/03_split.txt"
import glob, os, sys
ext, cal, tst, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
cases = sorted(os.path.basename(f)[:-9] for f in glob.glob(os.path.join(ext, '*_thin.npy')))
k = min(k, max(len(cases) - 10, 0))                      # never starve the reported set
stride = max(len(cases) // max(k, 1), 1)
cal_set = {cases[i] for i in range(0, len(cases), stride)[:k]} if k else set()
for c in cases:
    dst = cal if c in cal_set else tst                   # evenly spaced, reproducible
    for suf in ('_thin.npy', '_thick.npy', '_lung.npy', '_base.npy'):
        src = os.path.join(ext, c + suf)
        if os.path.exists(src):
            link = os.path.join(dst, c + suf)
            if not os.path.exists(link): os.symlink(os.path.abspath(src), link)
n_c = len(glob.glob(os.path.join(cal, '*_thin.npy')))
n_t = len(glob.glob(os.path.join(tst, '*_thin.npy')))
print(f'EXTERNAL+CAL: model from the internal cohort; external {len(cases)} cases '
      f'-> {n_c} calibration / {n_t} reported')
PYEOF
    CALROOT=$CAL
  else
    echo "   external design: TRAIN = $TAG ($N_TR cases), TEST = external ($N_TE cases)"
    echo "EXTERNAL: TRAIN = $TAG $N_TR cases / TEST = external $N_TE cases" > "$RES/03_split.txt"
    CALROOT=$TRAIN
  fi
elif [ "${NOSPLIT:-0}" = 1 ]; then
  echo "   !! NOSPLIT=1 — TRAIN = TEST = all cases; fine-tuned arms are IN-SAMPLE"
  TRAIN=$PAIRS; TEST=$PAIRS
  echo "NOSPLIT=1: TRAIN = TEST = all $(ls "$PAIRS"/*_thin.npy 2>/dev/null | wc -l) cases" \
       > "$RES/03_split.txt"
else
# Everything that is FITTED (protocol constants, forward operator, certificate
# thresholds, fine-tuned weights) is fitted on TRAIN. Everything REPORTED is
# measured on TEST. Same discipline as the public stage (85 train / 50 test).
# The split is by sorted anonymised case id, so it is reproducible and does not
# depend on directory order or on the run date.
if skip "$RES/03_split.txt"; then echo "   (already done)"; else
mkdir -p "$TRAIN" "$TEST"
$PY - "$PAIRS" "$TRAIN" "$TEST" <<'PYEOF' | tee "$RES/03_split.txt"
import glob, os, sys
pairs, tr, te = sys.argv[1:4]
cases = sorted(os.path.basename(f)[:-9] for f in glob.glob(os.path.join(pairs, '*_thin.npy')))
# every other case -> test; deterministic, balanced, and robust to small cohorts
for i, c in enumerate(cases):
    dst = te if i % 2 else tr
    for suf in ('_thin.npy', '_thick.npy', '_lung.npy'):
        src = os.path.join(pairs, c + suf)
        if os.path.exists(src):
            link = os.path.join(dst, c + suf)
            if not os.path.exists(link): os.symlink(os.path.abspath(src), link)
n_tr = len(glob.glob(os.path.join(tr, '*_thin.npy')))
n_te = len(glob.glob(os.path.join(te, '*_thin.npy')))
print(f'{len(cases)} cases -> TRAIN {n_tr} / TEST {n_te}')
print('TRAIN:', ' '.join(c for i, c in enumerate(cases) if i % 2 == 0))
print('TEST :', ' '.join(c for i, c in enumerate(cases) if i % 2 == 1))
PYEOF
fi
fi

echo "== [4/11] this cohort's protocol constants (effective width + tail relation)"
if skip "$PROTO"; then echo "   (already done)"; else
$PY scripts/fit_protocol_operator.py --root "${CALROOT:-$TRAIN}" --patch 40 64 64 \
    --protocol "$TAG/1mm-5mm/r5" --out "$PROTO" 2>&1 | tee "$RES/04_protocol.log"
fi

echo "== [5/11] learn this scanner's forward operator A_theta (halves the misfit)"
if skip "$FWDOP"; then echo "   (already done)"; else
$PY scripts/fit_forward_operator.py --root "${CALROOT:-$TRAIN}" --out "$FWDOP" \
    --epochs $([ "$MODE" = dryrun ] && echo 20 || echo 400) 2>&1 | tee "$RES/05_forward_op.log"
fi

echo "== [6/11] certificate NULL on a perfect reconstruction -> this cohort's thresholds"
if skip "$RES/cert_oracle.csv"; then echo "   (already done)"; else
$PY scripts/certify.py --root "${CALROOT:-$TRAIN}" --oracle --calibration "$PROTO" \
    --learned_op "$FWDOP" $MASK --csv "$RES/cert_oracle_train.csv" 2>&1 | tee "$RES/06_oracle.log"
$PY - "$RES/cert_oracle_train.csv" "$PROTO" <<'PYEOF' 2>&1 | tee -a "$RES/06_oracle.log"
import csv, json, sys, numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
g = lambda k: np.array([float(r[k]) for r in rows])
d, rs = g('delta_HU'), g('rho_struct')
s = np.array([float(r['s']) for r in rows if r['s'] != ''])
tau = lambda a: float(abs(np.mean(a)) + 2 * np.std(a))
T = dict(tau_delta=round(tau(d), 2), tau_rho=round(tau(rs), 3),
         tau_s=round(tau(s), 2) if len(s) else 2.0)
print('null on THIS cohort: delta %+.2f+-%.2f HU | rho_struct %.3f+-%.3f -> %s'
      % (d.mean(), d.std(), rs.mean(), rs.std(), T))
p = json.load(open(sys.argv[2])); p.update(T); json.dump(p, open(sys.argv[2], 'w'), indent=1)
PYEOF
# Now apply those thresholds to the HELD-OUT half. This is the honest null: if the
# certificate is calibrated, a perfect reconstruction of unseen scans still passes.
$PY scripts/certify.py --root "$TEST" --oracle --calibration "$PROTO" \
    --learned_op "$FWDOP" $MASK --csv "$RES/cert_oracle.csv" 2>&1 | tail -4 | tee -a "$RES/06_oracle.log"
fi

echo "== [7/11] THE HEADLINE: zero-shot, i.e. the deployment scenario"
# A township hospital has no paired 1 mm data, so the model it runs must have been
# trained somewhere else. That is this stage: public weights, applied to a scanner they
# have never seen. It needs no training, so it also front-loads the most important
# result - if the machine time runs out here, the primary claim is already measured.
CERT="--calibration $PROTO --learned_op $FWDOP $MASK --per_lobe"
if skip "$RES/cert_public_zeroshot.csv"; then echo "   (already done)"; else
$PY scripts/certify.py --root "$TEST" --ckpt public.pt $CERT \
    --csv "$RES/cert_public_zeroshot.csv" 2>&1 | tee "$RES/07_zeroshot.log"
echo "-- 生成级零样本"
$PY scripts/certify.py --root "$TEST" --ckpt public_flow.pt --flow --residual_scale $SR \
    --flow_steps $STEPS $CERT --csv "$RES/cert_flow_zeroshot.csv" 2>&1 | tee "$RES/07_flow_zeroshot.log"
echo "-- 经典非学习对照（构造上不可能平移）"
$PY scripts/baseline_deconv.py --root "$TEST" --calibration "$PROTO" \
    --csv "$RES/cert_landweber.csv" 2>&1 | tee "$RES/07_landweber.log"
echo "-- 单标量对照：几对配对就能复现一次微调？"
for N in 1 3 5 10; do
  $PY scripts/certify.py --root "$TEST" --ckpt public.pt $CERT \
      --bias_from "$TRAIN" --bias_pairs $N --csv "$RES/cert_biasonly_n$N.csv" \
      2>&1 | tee -a "$RES/07_biasonly.log"
done
fi

echo "-- 证书常数能否跨设备迁移？两种标定各跑一遍"
# The certificate's constants are fitted with paired 1 mm data, which the deployment
# site will not have. So test both: constants measured here (what the study can do) and
# constants carried over from the public cohort (what deployment would have to do).
if skip "$RES/cert_publiccal.csv"; then echo "   (already done)"; else
$PY scripts/certify.py --root "$TEST" --ckpt public_flow.pt --flow --residual_scale $SR \
    --flow_steps $STEPS --calibration runs/protocol_rplhr_5mm.json \
    --learned_op public_forward_op.pt $MASK \
    --csv "$RES/cert_publiccal.csv" 2>&1 | tee "$RES/07_publiccal.log"
fi

echo "== [7b/11] 零样本整卷评测（主结果）"
if skip "$RES/volume_zs_ours.csv"; then echo "   (already done)"; else
if [ ! -s public_cthnet.pt ]; then
  echo "!! public_cthnet.pt 缺失 —— 零样本骨干臂无法运行。"
  echo "   它是主结果所需，不是可选项。检查仓库是否完整 clone。"
else
  echo "-- 骨干零样本 + 缓存重建"
  $PY scripts/baseline_nets.py infer --net cthnet --root "$TEST" --ckpt public_cthnet.pt \
      --crop 256 --z_shift 2 --amp --grad_ckpt --n $NVOL --save_recon "$TEST" \
      --csv "$RES/volume_zs_cthnet.csv" 2>&1 | tail -9 | tee "$RES/07b_cthnet.log"
  echo "-- 我们的方法零样本（单采样，报指标用）"
  $PY scripts/infer_volume.py --root "$TEST" --ckpt public_flow.pt --flow --base_recon \
      --residual_scale $SR --flow_steps $STEPS --lung_blend 1 --samples 1 \
      --learned_op $FWDOP --calibration "$PROTO" --n $NVOL \
      --csv "$RES/volume_zs_ours.csv" 2>&1 | tail -9 | tee "$RES/07b_ours.log"
fi
echo "-- 插值对照"
$PY scripts/infer_volume.py --root "$TEST" --interp lanczos --n $NVOL \
    --csv "$RES/volume_lanczos.csv" 2>&1 | tail -6 | tee "$RES/07b_lanczos.log"
fi

echo "== [8/11] 可选：本地训练（仅在零样本不够好时才需要）"
# Stages 8-10 answer a different question - "how good could it be WITH local paired
# data" - which is not the deployment scenario. Run them only if stage 7b came out
# short; they cost 3-4 h of scarce on-site GPU time. Set LOCAL=1 to enable.
if [ "${LOCAL:-0}" != 1 ]; then
  echo "   跳过（LOCAL=1 启用）。零样本结果已在阶段 7/7b。"
else
echo "-- backbone: train CTHNet on this cohort, then cache its reconstructions"
# The generative stage models the residual a STRONG reconstruction leaves behind,
# so the backbone comes first and is then frozen. Caching its output lets the flow
# train on 64^3 crops even though CTHNet's patch embedding is locked to 256 in-plane.
if skip "$RES/08_backbone.log"; then echo "   (already done)"; else
$PY scripts/baseline_nets.py train --net cthnet --root "$TRAIN" --steps $BSTEPS \
    --crop 256 --z_shift 2 --amp --grad_ckpt --ckpt "$OUT/cthnet.pt" 2>&1 | tail -3 | tee "$RES/08_backbone.log"
for D in "$TRAIN" "$TEST"; do
  $PY scripts/baseline_nets.py infer --net cthnet --root "$D" --ckpt "$OUT/cthnet.pt" \
      --crop 256 --z_shift 2 --amp --grad_ckpt --save_recon "$D" $MASK_CSV \
      2>&1 | tail -3 | tee -a "$RES/08_backbone.log"
done
fi

echo "== [9/11] generative stage: flow-matched residual on the frozen backbone"
if skip "$OUT/flow.pt"; then echo "   (already done)"; else
$PY -m scte_r.train --data pairs --root "$TRAIN" --flow --base_recon \
    --residual_scale 0.028 --calibration "$PROTO" --clip 1.0 --ema 0.999 --lr 1e-3 \
    --epochs $EPOCHS --workers 2 --ckpt "$OUT/flow.pt" 2>&1 | tail -3 | tee "$RES/09_flow.log"
fi

echo "== [10/11] certificate on the held-out cohort"
if skip "$RES/cert_ours.csv"; then echo "   (already done)"; else
$PY scripts/certify.py --root "$TEST" --ckpt "$OUT/flow.pt" --flow --base_recon \
    --residual_scale 0.028 --flow_steps $STEPS $CERT --csv "$RES/cert_ours.csv" \
    2>&1 | tee "$RES/10_cert_ours.log"
echo "-- the deterministic control: does THIS scanner also produce a ~20 HU displacement?"
$PY -m scte_r.train --data pairs --root "$TRAIN" --arc --calibration "$PROTO" \
    --clip 1.0 --ema 0.999 --lr 1e-3 --epochs $EPOCHS --workers 2 \
    --ckpt "$OUT/det.pt" 2>&1 | tail -2 | tee "$RES/10_det.log"
$PY scripts/certify.py --root "$TEST" --ckpt "$OUT/det.pt" $CERT \
    --csv "$RES/cert_deterministic.csv" 2>&1 | tee -a "$RES/10_cert_ours.log"
fi

fi   # end of the optional local-training block

echo "== [11/11] 可选：本地训练模型的整卷评测"
if [ "${LOCAL:-0}" != 1 ]; then echo "   跳过（LOCAL=1 启用）"; else
if skip "$RES/volume_ours_s1.csv"; then echo "   (already done)"; else
$PY scripts/infer_volume.py --root "$TEST" --interp lanczos --n $NVOL \
    --csv "$RES/volume_lanczos.csv" 2>&1 | tail -6 | tee "$RES/11_lanczos.log"
$PY scripts/baseline_nets.py infer --net cthnet --root "$TEST" --ckpt "$OUT/cthnet.pt" \
    --crop 256 --z_shift 2 --amp --grad_ckpt --n $NVOL \
    --csv "$RES/volume_cthnet.csv" 2>&1 | tail -8 | tee "$RES/11_cthnet.log"
COMMON="--root $TEST --ckpt $OUT/flow.pt --flow --base_recon --residual_scale 0.028"
COMMON="$COMMON --flow_steps $STEPS --lung_blend 1 --learned_op $FWDOP --calibration $PROTO --n $NVOL"
echo "-- densitometry output (single sample)"
$PY scripts/infer_volume.py $COMMON --samples 1 --csv "$RES/volume_ours_s1.csv" \
    2>&1 | tail -8 | tee "$RES/11_ours_s1.log"
echo "-- viewing output (4-sample mean)"
$PY scripts/infer_volume.py $COMMON --samples 4 --csv "$RES/volume_ours_s4.csv" \
    2>&1 | tail -8 | tee "$RES/11_ours_s4.log"
fi

fi   # end of the optional stage-11 block

echo
: > "$RES/SUMMARY.txt"          # truncate: a previous dryrun in this tag would otherwise
                               # leave its "these numbers are meaningless" banner sitting
                               # above a real run's results
if [ "$MODE" = dryrun ]; then
  cat <<'WARN' | tee -a "$RES/SUMMARY.txt"

################################################################################
  这是 DRY RUN。下面每一个数字都不可解读，原因具体：

    * 采样步数 16 而非 64  -> 采样远未收敛，注入的是噪声不是结构，
                              我们这一臂的 PSNR 会低于插值、LAA 偏差会翻号
    * 骨干训练 30 步        -> 相当于随机权重
    * 证书阈值来自 3 例     -> |mean|+2sd 在 n=3 上极不稳定，
                              连完美重建都可能通不过（实测：同一模型
                              4 例零分布下 4/4，8 例下 0/8）

  dry run 只回答一件事：管路通不通。任何"效果好不好"的判断都要等正式跑。
################################################################################

WARN
fi
$PY - "$RES" <<'PYEOF' | tee -a "$RES/SUMMARY.txt"
import csv, glob, os, sys, numpy as np
R = sys.argv[1]
sp = os.path.join(R, '03_split.txt')
if os.path.exists(sp):
    head = open(sp).read().splitlines()[0]
    tail = ('- fine-tuned arms below are IN-SAMPLE' if head.startswith('NOSPLIT')
            else '- external validation: nothing below was fitted on the test cohort'
            if head.startswith('EXTERNAL')
            else '- fitted on TRAIN, every number below measured on TEST')
    print('==', head, tail, '\n')
def rd(f):
    p = os.path.join(R, f)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []
def g(rows, k): return np.array([float(r[k]) for r in rows if r.get(k) not in (None, '')])
print('== per-scan certificate (patch level)')
print(f"{'arm':22s} {'|delta| HU':>10s} {'LAA MAE':>8s} {'certified':>10s}")
for f, n in [('cert_oracle.csv','perfect recon (null)'), ('cert_landweber.csv','Landweber (control)'),
             ('cert_public_zeroshot.csv','public.pt zero-shot'),
             ('cert_flow_zeroshot.csv','flow zero-shot'),
             ('cert_publiccal.csv','flow, PUBLIC calibration'),
             ('cert_deterministic.csv','deterministic (local)'), ('cert_ours.csv','OURS (local)')]:
    r = rd(f)
    if not r: continue
    mae = np.mean(np.abs(g(r,'laa_rec') - g(r,'laa_ref'))) if 'laa_rec' in r[0] else float('nan')
    c = sum(x.get('verdict') == 'certified' for x in r)
    print(f"{n:22s} {np.mean(np.abs(g(r,'delta_HU'))):10.2f} {mae:8.2f} {c:6d}/{len(r)}")
print('\n== by emphysema severity (1 mm reference LAA-950 tertiles of the test cohort)')
# With no third cohort to compare against hospital-reported LAA, severity
# stratification is the main evidence that the method holds where it matters:
# a method that only works on near-normal lungs is not clinically useful.
r = rd('cert_ft_flow.csv'); r0 = rd('cert_public_zeroshot.csv')
if r and 'laa_ref' in r[0]:
    ref = g(r, 'laa_ref'); q1, q2 = np.percentile(ref, [33.3, 66.7])
    print(f"{'stratum':22s} {'n':>4s} {'ref LAA':>9s} {'FLOW MAE':>9s} {'zero-shot MAE':>14s}")
    hi_max = ref.max()
    for lo, hi, n in [(-1, q1, 'mild'), (q1, q2, 'moderate'), (q2, hi_max, 'severe')]:
        m = (ref > lo) & (ref <= hi)
        if not m.any(): continue
        f_mae = np.mean(np.abs(g(r, 'laa_rec')[m] - ref[m]))
        z = f'{np.mean(np.abs(g(r0, "laa_rec")[m] - ref[m])):14.2f}' if r0 and len(g(r0,'laa_rec')) == len(ref) else f'{"-":>14s}'
        print(f"{n + f' (<={hi:.2f}%)':22s} {m.sum():4d} {ref[m].mean():9.2f} {f_mae:9.2f} {z}")

print('\n== 零样本整卷（主结果：模型从未见过这台设备）')
print(f"{'arm':26s} {'PSNR':>7s} {'lungPSNR':>9s} {'SSIM':>7s} {'LAA950':>7s} {'LAA910':>7s} {'P15':>6s}")
for f, n in [('volume_lanczos.csv','Lanczos-3'), ('volume_zs_cthnet.csv','CTHNet zero-shot'),
             ('volume_zs_ours.csv','OURS zero-shot'),
             ('volume_cthnet.csv','CTHNet local-trained'),
             ('volume_ours_s1.csv','OURS local-trained')]:
    r = rd(f)
    if not r: continue
    e = lambda k: np.mean(np.abs(g(r, k+'_rec') - g(r, k+'_ref')))
    print(f"{n:26s} {g(r,'psnr').mean():7.2f} {g(r,'lung_psnr').mean():9.2f} "
          f"{g(r,'ssim').mean():7.4f} {e('laa950'):7.2f} {e('laa910'):7.2f} {e('p15'):6.2f}")
print('\n== whole-lung bias (recon - 1 mm reference); LAA<0 and Perc>0 both mean "reports LESS emphysema"')
print(f"{'arm':24s} {'LAA950':>8s} {'LAA910':>8s} {'Perc15':>8s} {'Perc10':>8s}")
for f, n in [('volume_lanczos.csv','Lanczos-3'), ('volume_zs_cthnet.csv','CTHNet zero-shot'),
             ('volume_zs_ours.csv','OURS zero-shot')]:
    r = rd(f)
    if not r: continue
    b = [np.mean(g(r,k+'_rec') - g(r,k+'_ref')) for k in ('laa950','laa910','p15','p10')]
    print(f"{n:24s} " + ' '.join(f'{x:+8.2f}' for x in b))
PYEOF
echo
echo "== DONE. Export ONLY: $RES/*.csv, *.log, SUMMARY.txt  (no image data, no patient"
echo "   identifiers). $PAIRS and ID_MAP_DO_NOT_EXPORT.csv stay on this machine."
du -sh "$PAIRS" "$RES" 2>/dev/null || true
