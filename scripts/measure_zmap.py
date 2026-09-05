"""Measure the affine z map between the two series: thin_index = slope * thick_index + c.

The differences between expected and actual position grow linearly with z (27->57->87 on
one case), which is not an offset at all - it is a wrong sampling ratio. prep_pairs is
hard-wired to --downsample 5; where the true ratio is 7 the error accumulates by 2
slices per thick slice, and by mid-volume the compared windows share nothing. Peak
correlations of 0.997 confirm the content matches perfectly; only the mapping was wrong.

Fits the slope per case so the real ratio can be read off, then reports how many cases
share each ratio.

  python scripts/measure_zmap.py --root PRIVATE/henan/pairs --n 20
"""
import argparse, csv, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--n', type=int, default=20)
ap.add_argument('--win', type=int, default=5, help='thin slices averaged per probe')
ap.add_argument('--probes', type=int, default=5)
ap.add_argument('--csv', default=None)
a = ap.parse_args()

files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
rows = []
for f in files:
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    D, nk = thin.shape[0], thick.shape[0]
    cs = np.concatenate([np.zeros((1,) + thin.shape[1:], np.float64), np.cumsum(thin, 0)])
    ks, js, cc = [], [], []
    for k in np.linspace(nk * 0.15, nk * 0.85, a.probes).astype(int):
        tk = thick[k].ravel(); tk = (tk - tk.mean()) / (tk.std() + 1e-8)
        best, bj = -2.0, -1
        for j in range(0, D - a.win + 1):
            w = ((cs[j + a.win] - cs[j]) / a.win).ravel().astype(np.float32)
            sd = w.std()
            if sd < 1e-6: continue
            c = float(((w - w.mean()) / sd * tk).mean())
            if c > best: best, bj = c, j
        ks.append(k); js.append(bj); cc.append(best)
    if len(ks) < 2: continue
    slope, icpt = np.polyfit(ks, js, 1)
    resid = float(np.max(np.abs(np.polyval([slope, icpt], ks) - np.array(js))))
    rows.append(dict(case=f[:-9], slope=round(float(slope), 3), intercept=round(float(icpt), 1),
                     fit_resid=round(resid, 1), peak_corr=round(float(np.median(cc)), 4),
                     thin=D, thick=nk))
    print(f"{f[:-9][:16]:<16s} slope {slope:6.3f}  intercept {icpt:7.1f}  "
          f"拟合残差 {resid:5.1f}  峰值相关 {np.median(cc):.3f}", flush=True)

s = np.array([r['slope'] for r in rows])
print(f"\n{len(rows)} 例  斜率：中位 {np.median(s):.3f}  范围 [{s.min():.3f}, {s.max():.3f}]")
u, n = np.unique(np.round(s).astype(int), return_counts=True)
print(f"取整后的分布：{dict(zip(u.tolist(), n.tolist()))}")
print(f"峰值相关中位 {np.median([r['peak_corr'] for r in rows]):.4f}"
      f"  （>0.99 说明内容匹配，只是映射错了）")
out = a.csv or os.path.join(a.root, 'zmap.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', out)
