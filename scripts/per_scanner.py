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
results = []
for gi, g in enumerate(runnable):
    tag = 'g%d' % gi
    gd = os.path.join(a.out, tag)
    for role, root in (('cal', a.cal), ('test', a.test)):
        sub = os.path.join(gd, role)
        os.makedirs(sub, exist_ok=True)
        for c in groups[g][role]:
            for suf in ('_thin.npy', '_thick.npy', '_lung.npy', '_base.npy'):
                src = os.path.join(root, c + suf)
                if os.path.exists(src):
                    link_or_copy(src, os.path.join(sub, c + suf))
    print(f'\n===== {tag}: {g}  (标定 {len(groups[g]["cal"])} / 上报 {len(groups[g]["test"])})')
    proto = os.path.join(gd, 'protocol.json')
    fwd = os.path.join(gd, 'forward_op.pt')
    cal, tst = os.path.join(gd, 'cal'), os.path.join(gd, 'test')

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
        # thresholds from this group's own null
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
                      ('flow', ['--ckpt', a.flow_ckpt, '--flow',
                                '--residual_scale', str(a.residual_scale),
                                '--flow_steps', str(a.flow_steps)])):
        out = os.path.join(gd, f'cert_{nm}.csv')
        if os.path.exists(out):
            continue
        print(f'  认证 {nm} ...', flush=True)
        run([PY, f'{ROOT}/scripts/certify.py', '--root', tst, '--calibration', proto,
             '--learned_op', fwd, '--lung_mask', '--csv', out] + extra, f'{nm}.log')

    row = dict(group=g, tag=tag, n_cal=len(groups[g]['cal']), n_test=len(groups[g]['test']),
               w=w)
    for nm in ('oracle_test', 'flow'):
        p = os.path.join(gd, f'cert_{nm}.csv')
        if os.path.exists(p):
            rr = list(csv.DictReader(open(p)))
            row[nm] = '%d/%d' % (sum(x['verdict'] == 'certified' for x in rr), len(rr))
    results.append(row)

print('\n\n===== 按机器分别标定后的结果')
print(f"{'组':<34s}{'有效层厚':>10s}{'标定':>6s}{'上报':>6s}{'Oracle':>10s}{'Flow':>10s}")
for r in results:
    print(f"{r['group'][:34]:<34s}{str(r['w']):>10s}{r['n_cal']:>6d}{r['n_test']:>6d}"
          f"{r.get('oracle_test', '-'):>10s}{r.get('flow', '-'):>10s}")
print('\n对照：混合标定下 Oracle 85.1%、Flow 12.6%、公开标定 2.9%（444 例）。')
print('若分组后 Flow 明显上升，说明此前的失败有相当部分来自"两台机器的物理被平均掉了"。')
