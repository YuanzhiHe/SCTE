"""Agreement analysis in the form clinical journals expect.

CCC alone does not satisfy a medical reviewer, and for good reason: it is a single
number that mixes bias with correlation, and it is inflated by between-subject spread -
measured here, the stratum with the SMALLEST errors had the LOWEST CCC. Bland-Altman
reports the two things a clinician needs separately: how far off the method is on
average (bias), and how far off it can be for one patient (limits of agreement).

Every statistic is reported with a bootstrap 95% CI over cases, because point estimates
on 50 scans invite exactly the objection they deserve.

  python scripts/agreement.py --arms AL_vol_lanczos:Lanczos AL_vol_cthnet:CTHNet \
      AL_final_s1:ours --metric laa950
"""
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument('--arms', nargs='+', required=True, metavar='CSV:LABEL')
ap.add_argument('--metric', default='laa950')
ap.add_argument('--unit', default='pp')
ap.add_argument('--boot', type=int, default=5000)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--csv', default=None)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    return 2 * ((x - mx) * (y - my)).mean() / (vx + vy + (mx - my) ** 2 + 1e-12)


def stats(ref, rec):
    d = rec - ref
    bias = d.mean()
    sd = d.std(ddof=1)
    return dict(bias=bias, loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
                mae=np.abs(d).mean(), rmse=np.sqrt((d ** 2).mean()), ccc=ccc(rec, ref),
                sd=sd)


def boot_ci(ref, rec, key):
    n = len(ref)
    vals = np.empty(a.boot)
    for b in range(a.boot):
        i = rng.integers(0, n, n)
        vals[b] = stats(ref[i], rec[i])[key]
    return np.percentile(vals, [2.5, 97.5])


rows = []
print(f'\n{a.metric} agreement vs the 1 mm reference, bootstrap 95% CI over cases '
      f'({a.boot} resamples)\n')
hdr = f"{'arm':<20s}{'n':>4s}{'bias (95% CI)':>26s}{'95% LoA':>22s}{'MAE':>16s}{'CCC':>20s}"
print(hdr); print('-' * len(hdr))
for spec in a.arms:
    fn, label = spec.split(':', 1)
    p = os.path.join('results', fn if fn.endswith('.csv') else fn + '.csv')
    if not os.path.exists(p):
        print(f'{label:<20s}  (missing {p})'); continue
    r = list(csv.DictReader(open(p)))
    kr, kc = a.metric + '_ref', a.metric + '_rec'
    if kr not in r[0]:
        kr, kc = 'laa_ref', 'laa_rec'                 # patch-level CSVs use short names
    ref = np.array([float(x[kr]) for x in r])
    rec = np.array([float(x[kc]) for x in r])
    s = stats(ref, rec)
    cb = boot_ci(ref, rec, 'bias'); cm = boot_ci(ref, rec, 'mae'); cc = boot_ci(ref, rec, 'ccc')
    print(f"{label:<20s}{len(ref):>4d}"
          f"{s['bias']:>10.2f} [{cb[0]:6.2f},{cb[1]:6.2f}]"
          f"{s['loa_lo']:>10.2f} to{s['loa_hi']:6.2f}"
          f"{s['mae']:>7.2f} [{cm[0]:.2f},{cm[1]:.2f}]"
          f"{s['ccc']:>8.3f} [{cc[0]:.3f},{cc[1]:.3f}]")
    rows.append(dict(arm=label, n=len(ref), metric=a.metric,
                     bias=round(s['bias'], 3), bias_lo=round(cb[0], 3), bias_hi=round(cb[1], 3),
                     loa_lo=round(s['loa_lo'], 3), loa_hi=round(s['loa_hi'], 3),
                     mae=round(s['mae'], 3), mae_lo=round(cm[0], 3), mae_hi=round(cm[1], 3),
                     rmse=round(s['rmse'], 3), ccc=round(s['ccc'], 4),
                     ccc_lo=round(cc[0], 4), ccc_hi=round(cc[1], 4)))

print(f"\nbias = mean(recon - reference), in {a.unit}. LoA = bias +- 1.96 SD: the interval "
      f"containing 95% of\nindividual patients' errors - the number that decides whether a "
      f"single scan can be trusted.")

# paired difference between arms: does the ranking survive resampling?
if len(rows) >= 2:
    print('\npaired comparison of |error| against the last arm '
          '(bootstrap 95% CI of the difference; negative favours the last arm)\n')
    specs = [s.split(':', 1) for s in a.arms]
    last_fn, last_lb = specs[-1]
    lp = os.path.join('results', last_fn if last_fn.endswith('.csv') else last_fn + '.csv')
    lr = {x['case']: x for x in csv.DictReader(open(lp))}
    kr, kc = a.metric + '_ref', a.metric + '_rec'
    if kr not in next(iter(lr.values())):
        kr, kc = 'laa_ref', 'laa_rec'
    for fn, label in specs[:-1]:
        p = os.path.join('results', fn if fn.endswith('.csv') else fn + '.csv')
        if not os.path.exists(p): continue
        o = list(csv.DictReader(open(p)))
        common = [x for x in o if x['case'] in lr]
        if not common: continue
        e1 = np.array([abs(float(x[kc]) - float(x[kr])) for x in common])
        e2 = np.array([abs(float(lr[x['case']][kc]) - float(lr[x['case']][kr])) for x in common])
        d = e2 - e1
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = '' if lo <= 0 <= hi else '  *'
        print(f'  {last_lb} vs {label:<16s} n={len(d):3d}  '
              f'delta |error| {d.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}]{sig}')
    print('\n  * = the 95% CI excludes zero')

if a.csv and rows:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('\nwrote', a.csv)
