"""Term-level AUC, computed WITHIN domain as well as pooled.

A pooled AUC over a cohort containing a domain the model handles and a domain it
does not is mostly a domain detector. It will look like a usable per-case signal
and stop being one the moment the cohort is homogeneous - which is the situation
at any single deployment site. So every AUC here is reported pooled AND per
stratum, with a bootstrap CI, and the pooled value is not interpretable on its own
whenever the strata disagree.

Two targets, because they answer different questions:
  flip  - does the reported endpoint cross a diagnostic cutoff it should not have?
  tail  - is this case in the worst decile of |endpoint error|?

  python scripts/term_auc.py --cert results/cert_flow_zeroshot.csv \
      --volume results/volume_zs_ours.csv --audit PRIVATE/hebei/pairs_external/test/pair_audit.csv
"""
import argparse, csv, math, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cert', required=True)
ap.add_argument('--volume', required=True)
ap.add_argument('--audit', default=None, help='pair_audit.csv, to stratify by --key')
ap.add_argument('--key', default='thin_model')
ap.add_argument('--metric', default='laa950')
ap.add_argument('--cutoffs', type=float, nargs='+', default=[10.0, 15.0])
ap.add_argument('--tail_q', type=float, default=90.0, help='percentile defining the error tail')
ap.add_argument('--boot', type=int, default=2000)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

TERMS = (('|delta|', 'delta_HU', True), ('rho_struct', 'rho_struct', False),
         ('|s|', 's', True))


def auc(score, label):
    """Mann-Whitney U / (n_pos n_neg), ties at 0.5."""
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order), float)
    s = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(s):                      # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(score, label):
    v = np.empty(a.boot)
    n = len(score)
    for b in range(a.boot):
        i = rng.integers(0, n, n)
        v[b] = auc(score[i], label[i])
    v = v[np.isfinite(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (float('nan'),) * 2


cert = {}
for r in csv.DictReader(open(a.cert)):
    c = r['case'][:-9] if r['case'].endswith('_thin.npy') else r['case']
    cert[c] = r
grp = {}
if a.audit and os.path.exists(a.audit):
    for r in csv.DictReader(open(a.audit)):
        grp[r['case']] = (r.get(a.key) or '?').strip()

rows = []
for r in csv.DictReader(open(a.volume)):
    c = r['case']
    if c not in cert:
        continue
    try:
        ref, rec = float(r[a.metric + '_ref']), float(r[a.metric + '_rec'])
    except (KeyError, ValueError):
        continue
    if not (math.isfinite(ref) and math.isfinite(rec)):
        continue
    t = {}
    for nm, col, ab in TERMS:
        v = cert[c].get(col, '')
        try:
            x = abs(float(v)) if ab else float(v)
        except (TypeError, ValueError):
            x = float('nan')
        t[nm] = x
    rows.append(dict(case=c, ref=ref, rec=rec, err=abs(rec - ref),
                     grp=grp.get(c, 'all'), **t))

if not rows:
    sys.exit('no usable shared cases')
print(f'\n{a.cert}  n={len(rows)}   分层依据 {a.key if grp else "（无，仅合并）"}')

strata = [('合并', rows)]
if grp:
    for g in sorted({r['grp'] for r in rows}):
        sub = [r for r in rows if r['grp'] == g]
        if len(sub) >= 30:
            strata.append((g, sub))

for tname, target in ([(f'翻转@{c:g}', ('flip', c)) for c in a.cutoffs]
                      + [(f'误差尾部(前{100-a.tail_q:g}%)', ('tail', None))]):
    print(f'\n=== 预测目标：{tname}')
    print(f"  {'分层':<16s}{'n':>5s}{'阳性':>6s}" + ''.join(f'{n:>22s}' for n, _, _ in TERMS))
    for sname, sub in strata:
        if target[0] == 'flip':
            cut = target[1]
            lab = np.array([(r['ref'] < cut <= r['rec']) or (r['rec'] < cut <= r['ref'])
                            for r in sub])
        else:
            e = np.array([r['err'] for r in sub])
            lab = e >= np.percentile(e, a.tail_q)
        if lab.sum() < 5 or (~lab).sum() < 5:
            print(f'  {sname:<16s}{len(sub):>5d}{int(lab.sum()):>6d}'
                  f'      阳性/阴性过少，不计算')
            continue
        line = f'  {sname:<16s}{len(sub):>5d}{int(lab.sum()):>6d}'
        for nm, _, _ in TERMS:
            sc = np.array([r[nm] for r in sub])
            ok = np.isfinite(sc)
            if ok.sum() < len(sc) * 0.8 or lab[ok].sum() < 5:
                line += f'{"不可用":>22s}'; continue
            A = auc(sc[ok], lab[ok]); lo, hi = auc_ci(sc[ok], lab[ok])
            star = '' if (lo <= 0.5 <= hi) else ' *'
            line += f'{f"{A:.3f} [{lo:.2f},{hi:.2f}]{star}":>22s}'
        print(line)

print('\n* = 自助法 95% CI 不含 0.5。')
print('读法：合并行高而各分层行落回 0.5，说明该项分辨的是「域」而不是「逐例误差」。')
print('单站点部署时队列是同质的，此时能用的只有分层行的数值。')
