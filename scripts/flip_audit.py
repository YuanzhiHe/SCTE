"""Does the certificate separate the cases whose DIAGNOSIS changes?

A certification rate says nothing on its own. What a township hospital needs is
narrower: among the scans the certificate vouches for, how often does the reported
LAA cross a diagnostic cutoff it should not have crossed - and is that rate lower
than among the scans it flags?

The comparison has to be against the base rate, not against zero. An arm that
certifies 98% of everything will also certify ~98% of the flips; that is not the
certificate failing, it is the certificate being uninformative, and the two call
for different fixes. So this prints the flip rate in each verdict stratum side by
side, with the count in each stratum, and refuses to claim discrimination when a
stratum is too small to support one.

Discarding flagged scans also discards good ones, so the trade is printed as well:
how much the flip rate drops, and how many cases are lost to get it.

  python scripts/flip_audit.py --cert results/cert_ours_local.csv \
      --volume results/volume_ours_local.csv --cutoffs 10 15
"""
import argparse, csv, math, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--cert', required=True, help='certificate CSV (verdict column)')
ap.add_argument('--volume', required=True, help='whole-volume CSV (the reported endpoint)')
ap.add_argument('--metric', default='laa950')
ap.add_argument('--cutoffs', type=float, nargs='+', default=[10.0, 15.0],
                help='diagnostic thresholds in the metric\'s own units (pp for LAA)')
ap.add_argument('--label', default=None)
a = ap.parse_args()

for p in (a.cert, a.volume):
    if not os.path.exists(p):
        sys.exit(f'missing {p}')

ver = {}
for r in csv.DictReader(open(a.cert)):
    c = r['case']
    if c.endswith('_thin.npy'):
        c = c[:-9]
    ver[c] = r['verdict']

rows = []
for r in csv.DictReader(open(a.volume)):
    c = r['case']
    if c not in ver:
        continue
    try:
        ref, rec = float(r[a.metric + '_ref']), float(r[a.metric + '_rec'])
    except (KeyError, ValueError):
        continue
    if not (math.isfinite(ref) and math.isfinite(rec)):
        continue
    rows.append((c, ver[c], ref, rec))

if not rows:
    sys.exit('no cases shared between the two CSVs (check the case id format)')

label = a.label or os.path.basename(a.cert)
print(f'\n===== {label}   n={len(rows)}  （{a.metric} 整卷值，判决取自证书）')
strata = sorted({v for _, v, _, _ in rows})
print('  判决分布  ' + '  '.join(
    f'{s} {sum(1 for _, v, _, _ in rows if v == s)}' for s in strata))

for cut in a.cutoffs:
    up = lambda ref, rec: ref < cut <= rec
    dn = lambda ref, rec: rec < cut <= ref
    flip = lambda ref, rec: up(ref, rec) or dn(ref, rec)
    tot = sum(flip(r, c) for _, _, r, c in rows)
    print(f'\n  --- 阈值 {cut:g}：整体翻转 {tot}/{len(rows)} ({100*tot/len(rows):.1f}%)'
          f'  上穿 {sum(up(r,c) for _,_,r,c in rows)} / 下穿 {sum(dn(r,c) for _,_,r,c in rows)}')
    print(f"    {'判决':<12s}{'例数':>6s}{'翻转':>8s}{'翻转率':>10s}")
    rate = {}
    for s in strata:
        sub = [(r, c) for _, v, r, c in rows if v == s]
        f = sum(flip(r, c) for r, c in sub)
        rate[s] = (f, len(sub))
        note = '' if len(sub) >= 20 else '   (例数过少，不足以判别)'
        print(f'    {s:<12s}{len(sub):>6d}{f:>8d}{100*f/max(len(sub),1):>9.1f}%{note}')

    cf, cn = rate.get('certified', (0, 0))
    ff, fn = rate.get('flagged', (0, 0))
    if cn >= 20 and fn >= 20:
        rr = (ff / fn) / max(cf / cn, 1e-9)
        print(f'    → 标记组的翻转率是认证组的 {rr:.2f} 倍'
              f'（{100*ff/fn:.1f}% vs {100*cf/cn:.1f}%）')
        kept = 100 * cf / cn
        print(f'    → 只上报认证例：翻转率 {100*tot/len(rows):.1f}% → {kept:.1f}%，'
              f'代价是丢弃 {len(rows)-cn}/{len(rows)} 例（{100*(len(rows)-cn)/len(rows):.0f}%）')
    elif fn < 20:
        print(f'    → 标记组只有 {fn} 例：证书在本臂上已饱和，判别力无法被检验'
              f'（既不能说有，也不能说没有）')

print('\n读法：证书有判别力 = 标记组的翻转率显著高于认证组。若认证率接近 100%，'
      '标记组会小到无法检验，\n此时"认证例中有 X 例翻转"只反映基础发生率，不构成证书失效的证据。')
