"""Fit the protocol constants per SCANNER instead of pooling them.

Measured on the live external cohort: 145 cases from one scanner and 349 from another,
and the z-spacing ratio splits exactly along that line - 5 for one model, 6-7 for the
other. Pooling them produces one effective width (2.75 mm here) that belongs to neither
machine, and the certificate is built on that width: with public constants 2.9% of scans
certify, with pooled local constants 12.6%, against an oracle ceiling of 85.1%.

Splits a prepared cohort by scanner, then fits and certifies each group on its own. Also
turns a single mixed external cohort into a two-scanner external validation, which is a
stronger design than the pooled one it replaces.

  python scripts/per_scanner.py --cal PRIVATE/hebei/pairs_external/cal \
      --test PRIVATE/hebei/pairs_external/test --out PRIVATE/hebei/by_scanner
"""
import argparse, csv, os, shutil, subprocess, sys, collections

ap = argparse.ArgumentParser()
ap.add_argument('--cal', required=True, help='calibration subset (constants are fitted here)')
ap.add_argument('--test', required=True, help='reported subset')
ap.add_argument('--out', required=True)
ap.add_argument('--audit', default=None,
                help='pair_audit.csv; defaults to the one beside --cal or --test')
ap.add_argument('--key', default='thin_model',
                help='metadata column to group by: thin_model, thin_kernel, or both '
                     'via thin_model+thin_kernel')
ap.add_argument('--min_cases', type=int, default=12,
                help='groups smaller than this cannot support a protocol fit (needs >=3 '
                     'usable cases) or a null distribution, and are reported not fitted')
ap.add_argument('--flow_ckpt', default='public_flow.pt')
ap.add_argument('--residual_scale', type=float, default=0.028)
ap.add_argument('--flow_steps', type=int, default=64)
ap.add_argument('--base_recon', action='store_true',
                help='certify the hybrid arm (frozen backbone + flow residual) instead of '
                     'flow on plain up-sampling. Needs <case>_base.npy, i.e. the backbone '
                     'pass must have run first.')
ap.add_argument('--suffix', default='', help='appended to output names, to keep a '
                                             '--base_recon run beside a plain one')
ap.add_argument('--pooled_control', action='store_true',
                help='also fit and certify ONE pooled calibration on the same cases, as a '
                     'control. Without it a rise in certification cannot be attributed to '
                     'per-machine constants, because the earlier pooled numbers were '
                     'produced before the z-spacing prep fix.')
ap.add_argument('--dry', action='store_true', help='only show the grouping')
a = ap.parse_args()

PY = sys.executable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def find_audit():
    if a.audit:
        return a.audit
    for d in (a.cal, a.test, os.path.dirname(a.cal.rstrip('/')),
              os.path.dirname(a.test.rstrip('/'))):
        p = os.path.join(d, 'pair_audit.csv')
        if os.path.exists(p):
            return p
    return None


audit = find_audit()
if not audit:
    sys.exit('pair_audit.csv not found; pass --audit. Without the scanner metadata this '
             'script has nothing to group by.')
meta = {}
for r in csv.DictReader(open(audit)):
    keys = a.key.split('+')
    val = ' | '.join((r.get(k) or '?').strip() for k in keys)
    meta[r['case']] = val

groups = collections.defaultdict(lambda: {'cal': [], 'test': []})
for role, root in (('cal', a.cal), ('test', a.test)):
    if not os.path.isdir(root):
        sys.exit(f'{root} does not exist')
    for f in sorted(os.listdir(root)):
        if not f.endswith('_thin.npy'):
            continue
        groups[meta.get(f[:-9], '?未知')][role].append(f[:-9])

print(f'按 {a.key} 分组（标定集 {a.cal}，测试集 {a.test}）\n')
print(f"{'组':<40s}{'标定':>7s}{'上报':>7s}{'可拟合':>8s}")
runnable = []
for g, d in sorted(groups.items(), key=lambda x: -len(x[1]['test'])):
    ok = len(d['cal']) >= max(a.min_cases, 3)
    print(f'{g[:40]:<40s}{len(d["cal"]):>7d}{len(d["test"]):>7d}{"是" if ok else "否":>8s}')
    if ok:
        runnable.append(g)
if not runnable:
    sys.exit('\n没有任何一组的标定集足够大。合并组，或降低 --min_cases。')
if a.dry:
    sys.exit(0)

os.makedirs(a.out, exist_ok=True)

# Each job is one calibration: a tag, a display name, and the two directories it is
# fitted and reported on. The per-machine jobs get a directory of links; the pooled
# control points straight at the originals, because copying ~500 volumes on a machine
# with no symlinks would cost more disk than the whole run.
jobs = []
if a.pooled_control:
    jobs.append(dict(tag='pooled', name='合并对照（修复后同一批数据）', cal=a.cal, test=a.test,
                     n_cal=sum(len(d['cal']) for d in groups.values()),
                     n_test=sum(len(d['test']) for d in groups.values()), link=False))
for gi, g in enumerate(runnable):
    jobs.append(dict(tag='g%d' % gi, name=g, group=g, link=True,
                     n_cal=len(groups[g]['cal']), n_test=len(groups[g]['test'])))

results = []
for job in jobs:
    tag = job['tag']
    gd = os.path.join(a.out, tag)
    os.makedirs(gd, exist_ok=True)
    if job['link']:
        for role, root in (('cal', a.cal), ('test', a.test)):
            sub = os.path.join(gd, role)
            os.makedirs(sub, exist_ok=True)
            for c in groups[job['group']][role]:
                for suf in ('_thin.npy', '_thick.npy', '_lung.npy', '_base.npy'):
                    s = os.path.join(root, c + suf)
                    if os.path.exists(s):
                        link_or_copy(s, os.path.join(sub, c + suf))
        cal, tst = os.path.join(gd, 'cal'), os.path.join(gd, 'test')
    else:
        cal, tst = job['cal'], job['test']
    print(f'\n===== {tag}: {job["name"]}  (标定 {job["n_cal"]} / 上报 {job["n_test"]})')
    proto = os.path.join(gd, 'protocol.json')
    fwd = os.path.join(gd, 'forward_op.pt')

    def run(cmd, log):
        with open(os.path.join(gd, log), 'w') as fh:
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        return r.returncode

    if not os.path.exists(proto):
        print('  拟合协议常数 ...', flush=True)
        if run([PY, f'{ROOT}/scripts/fit_protocol_operator.py', '--root', cal,
                '--patch', '40', '64', '64', '--protocol', tag, '--out', proto],
               'protocol.log'):
            print('  !! 协议拟合失败，见 protocol.log'); continue
    w = '?'
    try:
        import json
        w = json.load(open(proto)).get('protocol_w', '?')
    except Exception:
        pass
    print(f'  有效层厚 {w} mm')

    if not os.path.exists(fwd):
        print('  拟合前向算子 ...', flush=True)
        run([PY, f'{ROOT}/scripts/fit_forward_operator.py', '--root', cal, '--out', fwd,
             '--epochs', '400'], 'forward_op.log')

    oc = os.path.join(gd, 'cert_oracle.csv')
    if not os.path.exists(oc):
        print('  零分布 → 阈值 ...', flush=True)
        run([PY, f'{ROOT}/scripts/certify.py', '--root', cal, '--oracle',
             '--calibration', proto, '--learned_op', fwd, '--lung_mask', '--csv', oc],
            'oracle.log')
        # thresholds from this job's own null
        try:
            import numpy as np, json
            rows = list(csv.DictReader(open(oc)))
            g_ = lambda k: np.array([float(x[k]) for x in rows if x.get(k) not in (None, '')])
            tau = lambda v: float(abs(v.mean()) + 2 * v.std()) if len(v) else 2.0
            p = json.load(open(proto))
            p.update(tau_delta=round(tau(g_('delta_HU')), 2),
                     tau_rho=round(tau(g_('rho_struct')), 3),
                     tau_s=round(tau(g_('s')), 2))
            json.dump(p, open(proto, 'w'), indent=1)
            print(f"  阈值 {p['tau_delta']=} {p['tau_rho']=} {p['tau_s']=}")
        except Exception as e:
            print('  阈值写入失败:', e)

    for nm, extra in (('oracle_test', ['--oracle']),
                      ('flow' + a.suffix,
                       ['--ckpt', a.flow_ckpt, '--flow',
                        '--residual_scale', str(a.residual_scale),
                        '--flow_steps', str(a.flow_steps)]
                       + (['--base_recon'] if a.base_recon else []))):
        out = os.path.join(gd, f'cert_{nm}.csv')
        if os.path.exists(out):
            continue
        print(f'  认证 {nm} ...', flush=True)
        run([PY, f'{ROOT}/scripts/certify.py', '--root', tst, '--calibration', proto,
             '--learned_op', fwd, '--lung_mask', '--csv', out] + extra, f'{nm}.log')

    row = dict(group=job['name'], tag=tag, n_cal=job['n_cal'], n_test=job['n_test'], w=w)
    for nm in ('oracle_test', 'flow' + a.suffix):
        p = os.path.join(gd, f'cert_{nm}.csv')
        if os.path.exists(p):
            rr = list(csv.DictReader(open(p)))
            row['flow' if nm.startswith('flow') else nm] = '%d/%d' % (
                sum(x['verdict'] == 'certified' for x in rr), len(rr))
    results.append(row)

print('\n\n===== 按机器分别标定后的结果')
print(f"{'组':<34s}{'有效层厚':>10s}{'标定':>6s}{'上报':>6s}{'Oracle':>10s}{'Flow':>10s}")
for r in results:
    print(f"{r['group'][:34]:<34s}{str(r['w']):>10s}{r['n_cal']:>6d}{r['n_test']:>6d}"
          f"{r.get('oracle_test', '-'):>10s}{r.get('flow', '-'):>10s}")

if a.pooled_control:
    print('\n读法：把每台机器的 Flow 与 pooled 行比，两者之差才是"分机器标定"的效果。')
    print('不要拿它和历史上那个 12.6% 比 —— 那个数是在 z 间距修复之前拟合的，差异里')
    print('混着 prep 的改动。若各机器与 pooled 基本持平，结论是一套标定跨厂商通用，')
    print('这对部署是更强的结论，不是失败。')
else:
    print('\n没有跑合并对照（--pooled_control）。缺了它，认证率的任何变化都无法归因于')
    print('分机器标定：历史上那个混合标定的数字是在 z 间距修复之前拟的。')
