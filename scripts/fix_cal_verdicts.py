"""Re-derive the verdict column of an already-written calibration null.

per_scanner.py ran the null BEFORE its thresholds existed, so those verdicts came
from the Arbiter's defaults (2.0 / 0.15 / 2.0) and describe nothing. The term
columns (delta_HU, rho_struct, s) do not depend on the thresholds, so the verdicts
can be rebuilt in place without re-running anything on the GPU.

Worth doing rather than ignoring: the calibration set's own coverage is the one
number that says whether the thresholds fit the data they were fitted on. It
should sit near 95%; if it does and the test set does not, the calibration subset
is not exchangeable with the test set, which is a sampling problem, not a
threshold-formula problem.

  python scripts/fix_cal_verdicts.py --cert PRIVATE/hebei/by_scanner/g0/cert_oracle.csv \
      --tau PRIVATE/hebei/by_scanner/g0/protocol.json
"""
import argparse, csv, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument('--cert', required=True)
ap.add_argument('--tau', required=True)
ap.add_argument('--out', default=None, help='default: overwrite --cert')
a = ap.parse_args()

p = json.load(open(a.tau))
for k in ('tau_delta', 'tau_rho', 'tau_s'):
    if k not in p:
        sys.exit(f'{a.tau} has no {k} — this is not the fitted calibration file')
rows = list(csv.DictReader(open(a.cert)))
if not rows:
    sys.exit(f'{a.cert} is empty')

n_cert = 0
for r in rows:
    if r['verdict'] == 'abstain':
        continue
    sv = r.get('s', '')
    viol = [(n_, q / t_) for n_, q, t_ in (
        ('displacement', abs(float(r['delta_HU'])), p['tau_delta']),
        ('structural', float(r['rho_struct']), p['tau_rho']),
        ('tail_recovery', abs(float(sv)) if sv not in ('', None) else 0.0, p['tau_s']))
        if q > t_]
    r['verdict'] = 'flagged' if viol else 'certified'
    r['violated'] = max(viol, key=lambda t: t[1])[0] if viol else ''
    n_cert += not viol

out = a.out or a.cert
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'{out}: 用 tau_delta={p["tau_delta"]} tau_rho={p["tau_rho"]} tau_s={p["tau_s"]} '
      f'重算 → certified {n_cert}/{len(rows)} ({100 * n_cert / len(rows):.1f}%)')
print('阈值按 |均值|+2sd 从这批数据自己拟的，所以这个数应当在 95% 附近。明显低于它，'
      '说明零分布不是高斯的；测试集比它低更多，则是标定子集不代表测试集。')
