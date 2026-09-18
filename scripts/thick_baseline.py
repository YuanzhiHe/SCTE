"""What the endpoints read if you measure the 5 mm series directly.

This is the row every reader wants first and the one the tables were missing:
before any reconstruction, how far off is densitometry on the acquired thick
series? Every other arm is an improvement over THIS, not over interpolation.

The lung set has to move to the thick grid, since that is where the voxels are.
Each thick voxel covers r thin voxels, so it is counted as lung when at least
--frac of them are; the default 0.5 is the majority rule. --frac 1.0 (every
sub-voxel lung) is the conservative variant and is worth running once as a
sensitivity check - it shrinks the set at the pleural surface, which is where
partial-volume averaging does most of its damage to the low-density tail.

Output columns match infer_volume.py, so the result drops straight into
common_subset.py and agreement.py as one more arm.

  python scripts/thick_baseline.py --root PRIVATE/hebei/pairs_external/test \
      --csv PRIVATE/hebei/results/volume_thick5mm.csv
"""
import argparse, csv, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True, help='prepared pairs (<case>_thin/_thick/_lung.npy)')
ap.add_argument('--downsample', type=int, default=5, help='thin slices per thick slice')
ap.add_argument('--frac', type=float, default=0.5,
                help='a thick voxel is lung when this fraction of its sub-voxels is')
ap.add_argument('--n', type=int, default=0)
ap.add_argument('--csv', default=None)
a = ap.parse_args()
r = a.downsample

LOBES = ['UL_L', 'LL_L', 'UL_R', 'ML_R', 'LL_R']


def endpoints(vol, mask):
    v = vol[mask]
    if v.size < 1000:
        return None
    return dict(laa950=100.0 * float((v < -950).mean()),
                laa910=100.0 * float((v < -910).mean()),
                p15=float(np.percentile(v, 15)),
                p10=float(np.percentile(v, 10)))


cases = sorted(f[:-9] for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.n:
    cases = cases[:a.n]
rows = []
for i, c in enumerate(cases, 1):
    fp = lambda suf: os.path.join(a.root, c + suf)
    if not all(os.path.exists(fp(s)) for s in ('_thin.npy', '_thick.npy', '_lung.npy')):
        continue
    thin = np.load(fp('_thin.npy')).astype(np.float32)
    thick = np.load(fp('_thick.npy')).astype(np.float32)
    lobe = np.load(fp('_lung.npy'))
    lung = lobe > 0
    D = thick.shape[0] * r
    if lung.shape[0] < D:                       # trim to whole slabs both ways
        D = (lung.shape[0] // r) * r
    thickD = D // r
    # fraction of each thick voxel's sub-voxels that are lung
    frac = lung[:D].reshape(thickD, r, *lung.shape[1:]).mean(axis=1)
    m_thick = frac >= a.frac
    m_thin = lung[:D]
    ref = endpoints(thin[:D], m_thin)
    rec = endpoints(thick[:thickD], m_thick)
    if ref is None or rec is None:
        print(f'[skip] {c}: lung set too small ({"thin" if ref is None else "thick"})')
        continue
    row = dict(case=c, lung_voxels=int(m_thin.sum()), thick_lung_voxels=int(m_thick.sum()))
    for k in ('laa950', 'laa910', 'p15', 'p10'):
        row[k + '_ref'] = ref[k]; row[k + '_rec'] = rec[k]
    for li, ln in enumerate(LOBES, 1):
        sl_thin = lobe[:D] == li
        fr = (lobe[:D] == li).reshape(thickD, r, *lobe.shape[1:]).mean(axis=1)
        sl_thick = fr >= a.frac
        row['laa950_ref_' + ln] = (100.0 * float((thin[:D][sl_thin] < -950).mean())
                                   if sl_thin.sum() > 1000 else float('nan'))
        row['laa950_rec_' + ln] = (100.0 * float((thick[:thickD][sl_thick] < -950).mean())
                                   if sl_thick.sum() > 1000 else float('nan'))
    rows.append(row)
    print('[%3d/%d] %-14s LAA950 1mm %6.2f%%  5mm %6.2f%%  (%+.2f)  P15 %+7.1f HU'
          % (i, len(cases), c, ref['laa950'], rec['laa950'],
             rec['laa950'] - ref['laa950'], rec['p15'] - ref['p15']), flush=True)

if not rows:
    sys.exit('no usable cases')

print(f'\n===== 直接在 5 mm 序列上测量（n={len(rows)}，肺体素判定阈值 frac={a.frac}）')
print(f"{'指标':<10s}{'1mm 参考':>12s}{'5mm 直测':>12s}{'偏差':>10s}{'MAE':>9s}"
      f"{'95% 一致性界':>22s}{'CCC':>8s}")
for k, u in (('laa950', 'pp'), ('laa910', 'pp'), ('p15', 'HU'), ('p10', 'HU')):
    ref = np.array([x[k + '_ref'] for x in rows])
    rec = np.array([x[k + '_rec'] for x in rows])
    d = rec - ref
    sd = d.std(ddof=1)
    mx, my = rec.mean(), ref.mean()
    ccc = 2 * ((rec - mx) * (ref - my)).mean() / (rec.var() + ref.var() + (mx - my) ** 2 + 1e-12)
    print(f'{k:<10s}{ref.mean():>12.2f}{rec.mean():>12.2f}{d.mean():>10.2f}'
          f'{np.abs(d).mean():>9.2f}'
          f'{f"{d.mean()-1.96*sd:.2f} ~ {d.mean()+1.96*sd:.2f}":>22s}{ccc:>8.3f}  {u}')

print('\n这一行是所有重建臂的真正对照：不重建、直接读 5 mm 序列会得到什么。')
print('注意肺体素集合在两个网格上不同（5 mm 的体素更大、更少），这正是该基线的性质，')
print('不是口径错误——临床上就是这样测的。用 --frac 1.0 再跑一次可给出保守变体。')

if a.csv:
    os.makedirs(os.path.dirname(os.path.abspath(a.csv)) or '.', exist_ok=True)
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote', a.csv)
