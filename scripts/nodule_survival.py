"""Per-nodule survival: did the lesion itself make it through the thick series?

The FROC attempt before this one was uninformative: the crude candidate generator hit
only 35% of nodules even on the REAL 1 mm reference, because nodules attached to
vessels get swallowed into a component whose centroid lands on the vessel. A detector
that bad hides whatever the reconstruction does.

So drop the detector. Every nodule has an annotated centre and diameter, so measure the
lesion where it is known to be, on each arm, and compare against the same measurement on
the 1 mm reference:

  peak / mean HU inside the annotated sphere - is the lesion still dense?
  contrast against a surrounding shell      - is it still distinguishable from lung?
  suprathreshold volume                     - is it still the same size?

'Detectable' is then a property of the image, not of my code: contrast above a margin.
"""
import argparse, csv, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='DATA/luna_pairs')
ap.add_argument('--recon_dir', default=None); ap.add_argument('--label', default='ours')
ap.add_argument('--n', type=int, default=0)
ap.add_argument('--margin', type=float, default=100., help='HU contrast to count as detectable')
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--csv', default=None)
a = ap.parse_args()
op = SSPForwardOperator(slice_fwhm_mm=5.0, downsample=a.downsample)

nod = {}
for r in csv.DictReader(open(os.path.join(a.root, 'nodules.csv'))):
    nod.setdefault(r['seriesuid'], []).append(r)


def sphere(vol, z, y, x, rz, ry, rx):
    zz = slice(max(int(z - rz), 0), int(z + rz) + 1)
    yy = slice(max(int(y - ry), 0), int(y + ry) + 1)
    xx = slice(max(int(x - rx), 0), int(x + rx) + 1)
    sub = vol[zz, yy, xx]
    if sub.size == 0:
        return None, None
    gz, gy, gx = np.ogrid[zz.start - z:zz.stop - z, yy.start - y:yy.stop - y,
                          xx.start - x:xx.stop - x]
    d = (gz / max(rz, .5)) ** 2 + (gy / max(ry, .5)) ** 2 + (gx / max(rx, .5)) ** 2
    return sub, d


rows = []
files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.n: files = files[:a.n]
for f in files:
    uid = f[:-9]
    if uid not in nod: continue
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    up = op.upsample_to_grid(torch.from_numpy(thick / 1000.)[None, None],
                             thin.shape[0])[0, 0].numpy() * 1000.
    arms = {'ref': thin, 'up': up}
    if a.recon_dir:
        rp = os.path.join(a.recon_dir, f.replace('_thin', '_rec'))
        if os.path.exists(rp): arms[a.label] = np.load(rp).astype(np.float32)
    for r in nod[uid]:
        z, y, x, dia = float(r['z']), float(r['y']), float(r['x']), float(r['diameter_mm'])
        sy, sx = float(r['sy']), float(r['sx'])
        rz, ry, rx = dia / 2, dia / 2 / sy, dia / 2 / sx
        rec = dict(uid=uid, diameter_mm=round(dia, 2))
        for k, v in arms.items():
            sub, d = sphere(v, z, y, x, rz, ry, rx)
            if sub is None: continue
            inside = d <= 1.0
            shell = (d > 1.44) & (d <= 4.0)          # 1.2x - 2x radius
            if inside.sum() < 3 or shell.sum() < 10: continue
            rec[k + '_peak'] = float(sub[inside].max())
            rec[k + '_mean'] = float(sub[inside].mean())
            rec[k + '_contrast'] = float(sub[inside].mean() - np.median(sub[shell]))
        if 'ref_peak' in rec: rows.append(rec)

A = lambda k: np.array([r[k] for r in rows if k in r], float)
arm_keys = ['up'] + ([a.label] if a.recon_dir else [])
print(f"\nLUNA16 逐结节存活（{len(set(r['uid'] for r in rows))} 例，{len(rows)} 个标注结节）\n")
print(f"{'臂':<18s}{'峰值HU':>9s}{'均值HU':>9s}{'对比度HU':>10s}{'对比度保留':>11s}{'可检出率':>10s}")
ref_c = A('ref_contrast')
print(f"{'真实1mm(真值)':<18s}{A('ref_peak').mean():9.0f}{A('ref_mean').mean():9.0f}"
      f"{ref_c.mean():10.0f}{'100%':>11s}{100*np.mean(ref_c > a.margin):9.1f}%")
for k in arm_keys:
    c = A(k + '_contrast'); nm = {'up': '5mm 上采样(现状)'}.get(k, k)
    print(f"{nm:<18s}{A(k+'_peak').mean():9.0f}{A(k+'_mean').mean():9.0f}"
          f"{c.mean():10.0f}{100*c.mean()/ref_c.mean():10.0f}%{100*np.mean(c > a.margin):9.1f}%")
d = np.array([r['diameter_mm'] for r in rows])
print(f"\n按结节直径分层的可检出率（对比度 > {a.margin:.0f} HU）")
print(f"{'直径':<12s}{'n':>5s}{'真实1mm':>9s}" + ''.join(f"{ {'up':'5mm上采样'}.get(k,k):>12s}" for k in arm_keys))
for lo, hi, nm in [(0, 6, '<6mm'), (6, 10, '6-10mm'), (10, 99, '>=10mm')]:
    m = (d >= lo) & (d < hi)
    if m.sum() < 2: continue
    line = f"{nm:<12s}{m.sum():5d}{100*np.mean(ref_c[m] > a.margin):8.1f}%"
    for k in arm_keys:
        line += f"{100*np.mean(A(k+'_contrast')[m] > a.margin):11.1f}%"
    print(line)
if a.csv:
    with open(a.csv, 'w', newline='') as fh:
        ks = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=ks, restval=''); w.writeheader(); w.writerows(rows)
    print('wrote', a.csv)
