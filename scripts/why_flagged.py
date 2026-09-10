"""A certification RATE is not a diagnosis. This says which term is binding.

`certified` needs all three of |delta| <= tau_delta, rho_struct <= tau_rho and
|s| <= tau_s to hold, and the CSV's `violated` column names only the term that
overshot its threshold by the largest factor. So a rate of 1.9% is compatible
with two very different worlds: a reconstruction sitting 20 HU away from what
the thick observation implies (the displacement mechanism this project is
about), or a threshold fitted on a null that does not transfer to the test set.
Those call for opposite fixes.

For each arm this prints the verdict split, how many cases each term rejects on
its own, and how far the term sits from its threshold. Comparing arms fitted on
different calibration cohorts also needs the ORACLE arm alongside: the oracle IS
the reference, so it should certify at roughly the coverage the thresholds were
built for (|mean| + 2 sd -> about 95%). An oracle well below that is a
calibration transfer failure, and every model rate under it inherits the problem.

  python scripts/why_flagged.py --cert PRIVATE/hebei/by_scanner/g0/cert_oracle_test.csv:SIEMENS-oracle \
      PRIVATE/hebei/by_scanner/g0/cert_flow.csv:SIEMENS-flow --tau PRIVATE/hebei/by_scanner/g0/protocol.json
"""
import argparse, csv, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cert', nargs='+', required=True, metavar='CSV[:LABEL]')
ap.add_argument('--tau', default=None,
                help='protocol.json holding tau_delta/tau_rho/tau_s. Without it the '
                     'thresholds are inferred from the arm itself and marked inferred.')
a = ap.parse_args()

tau = dict(tau_delta=2.0, tau_rho=0.15, tau_s=2.0)
src = 'default'
if a.tau and os.path.exists(a.tau):
    tau.update({k: v for k, v in json.load(open(a.tau)).items() if k in tau})
    src = a.tau
print(f'\n阈值来源 {src}: tau_delta={tau["tau_delta"]} HU  tau_rho={tau["tau_rho"]}  '
      f'tau_s={tau["tau_s"]}')

TERMS = (('displacement', 'delta_HU', 'tau_delta', True),
         ('structural', 'rho_struct', 'tau_rho', False),
         ('tail_recovery', 's', 'tau_s', True))

for spec in a.cert:
    fn, _, label = spec.partition(':')
    label = label or os.path.basename(fn)
    if not os.path.exists(fn):
        print(f'\n{label}: 缺文件 {fn}'); continue
    rows = list(csv.DictReader(open(fn)))
    if not rows:
        print(f'\n{label}: 空文件'); continue
    n = len(rows)
    ver = {}
    for r in rows:
        ver[r['verdict']] = ver.get(r['verdict'], 0) + 1
    cert = ver.get('certified', 0)
    print(f'\n===== {label}  n={n}')
    print('  判决  ' + '  '.join(f'{k} {v} ({100*v/n:.1f}%)' for k, v in sorted(ver.items())))

    # how many cases each term rejects ON ITS OWN, not just as the largest violator
    print(f"  {'项':<16s}{'单独否决':>10s}{'中位数':>12s}{'90分位':>12s}{'阈值':>10s}")
    solo = {}
    for name, col, tk, absolute in TERMS:
        v = np.array([float(r[col]) for r in rows if r.get(col) not in (None, '')], float)
        v = v[np.isfinite(v)]
        if not len(v):
            print(f'  {name:<16s}{"不可用":>10s}'); continue
        q = np.abs(v) if absolute else v
        rej = int((q > tau[tk]).sum())
        solo[name] = rej
        print(f'  {name:<16s}{rej:>7d} ({100*rej/n:4.1f}%){np.median(q):>12.3f}'
              f'{np.percentile(q, 90):>12.3f}{tau[tk]:>10.3f}')
    if solo:
        worst = max(solo, key=solo.get)
        print(f'  → 绑定项是 {worst}（单独就否决了 {100*solo[worst]/n:.1f}% 的病例）')

    d = np.array([float(r['delta_HU']) for r in rows], float)
    d = d[np.isfinite(d)]
    if len(d):
        print(f'  delta 有符号: 均值 {d.mean():+.2f} HU  中位数 {np.median(d):+.2f}  '
              f'sd {d.std():.2f}')
        if abs(d.mean()) > 2 * d.std() / max(len(d) ** 0.5, 1):
            print('    （均值明显偏离 0：是整体平移，不是随机误差 —— 一个标量改正就能移动它）')

print('\n读法：oracle 臂的认证率就是这套阈值在本测试集上的实际覆盖率。阈值按 |均值|+2sd')
print('构造，本应在 95% 附近；明显低于它说明标定集的零分布迁移不到测试集，此时任何模型')
print('臂的率都继承了这个问题，跨标定方案比较率是没有意义的。')
