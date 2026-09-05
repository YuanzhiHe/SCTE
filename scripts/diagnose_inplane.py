"""Do the two series share the same in-plane geometry?

Slice-mean profiles that correlate at 0.83-0.97 while the voxel-wise residual sits near
1.0 means the same anatomy is present in z but the volumes do not line up in x/y. FOV,
reconstruction centre or pixel spacing differ, and prep_pairs applies the THIN series'
lung bounding box to BOTH volumes - so a mismatch there crops different anatomy out of
each and produces a pair that is quietly meaningless.

Measures the in-plane shift directly by cross-correlating a slab-averaged thin slice
against its thick counterpart.

  python scripts/diagnose_inplane.py --root PRIVATE/henan/pairs --n 12
"""
import argparse, csv, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--n', type=int, default=12)
ap.add_argument('--downsample', type=int, default=5)
a = ap.parse_args()
r = a.downsample


def xy_shift(a2, b2):
    """Peak of the normalised cross-power spectrum: sub-pixel-free but robust."""
    A = np.fft.fft2(a2 - a2.mean()); B = np.fft.fft2(b2 - b2.mean())
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    c = np.fft.ifft2(R).real
    j = np.unravel_index(np.argmax(c), c.shape)
    dy = j[0] - (c.shape[0] if j[0] > c.shape[0] // 2 else 0)
    dx = j[1] - (c.shape[1] if j[1] > c.shape[1] // 2 else 0)
    return dy, dx, float(c.max())


meta = {}
mp = os.path.join(a.root, 'pair_audit.csv')
if os.path.exists(mp):
    for row in csv.DictReader(open(mp)):
        meta[row['case']] = row

files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
print(f"{'case':<14s}{'面内形状':>12s}{'dy':>5s}{'dx':>5s}{'峰值':>7s}"
      f"{'相关':>7s}{'薄FOV':>7s}{'厚FOV':>7s}{'判定':>10s}")
bad = 0
for f in files:
    uid = f[:-9]
    thin = np.load(os.path.join(a.root, f), mmap_mode='r')
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')), mmap_mode='r')
    k = thick.shape[0] // 2
    tk = np.asarray(thick[k], np.float32)
    tn = np.asarray(thin[k * r:(k + 1) * r], np.float32).mean(0)
    dy, dx, pk = xy_shift(tn, tk)
    cc = float(np.corrcoef(tn.ravel(), tk.ravel())[0, 1])
    m = meta.get(uid, {})
    fd_t, fd_k = m.get('thin_recon_diameter_mm', '?'), m.get('thick_recon_diameter_mm', '?')
    v = ('FOV 不同' if fd_t != fd_k and '?' not in (fd_t, fd_k)
         else '面内平移' if (abs(dy) > 2 or abs(dx) > 2)
         else '面内不符' if cc < 0.9 else 'OK')
    if v != 'OK': bad += 1
    print(f"{uid[:14]:<14s}{str(tuple(thin.shape[1:])):>12s}{dy:5d}{dx:5d}{pk:7.3f}"
          f"{cc:7.3f}{str(fd_t)[:6]:>7s}{str(fd_k)[:6]:>7s}{v:>10s}")
print(f"\n{bad}/{len(files)} 例面内不匹配")
print("\n相关 = 中间那一层，厚层 vs 对应 5 层薄层的平均。它们本应几乎相同（公开数据 >0.99）。")
print("dy/dx = 面内平移（体素）。prep_pairs 把薄层的肺包围盒同时用在两个体积上，")
print("所以面内几何一旦不同，裁出来的就是不同的解剖。")
