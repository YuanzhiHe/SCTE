"""For each thick slice, find WHERE in the thin volume it actually comes from.

Everything inferred so far has been wrong: separate acquisitions (times match),
a sub-slab z offset (realignment did nothing), differing in-plane geometry (headers are
identical). So stop inferring. For a handful of thick slices, correlate against a
5-slice average taken at EVERY position in the thin volume and report where the peak is.

  peak at 5k, high        -> aligned; the problem is not correspondence at all
  peak far from 5k        -> a large z offset both searches were blind to
  no peak anywhere        -> the two volumes really do not share content

  python scripts/locate_slab.py --root PRIVATE/henan/pairs --n 4
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--n', type=int, default=4)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--probes', type=int, default=3)
ap.add_argument('--cases', default=None, help='comma-separated case ids to force')
a = ap.parse_args()
r = a.downsample

files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.cases:
    want = set(a.cases.split(','))
    files = [f for f in files if f[:-9] in want] or files[:a.n]
else:
    files = files[:a.n]

for f in files:
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    D, nk = thin.shape[0], thick.shape[0]
    # cumulative sum lets every 5-slice window be formed in O(1)
    cs = np.concatenate([np.zeros((1,) + thin.shape[1:], np.float64), np.cumsum(thin, 0)])
    print(f"\n=== {f[:-9][:16]}   thin {D}  thick {nk}")
    print(f"{'厚层层号':>8s}{'期望位置':>9s}{'实测最佳':>9s}{'差值':>7s}{'峰值相关':>9s}{'期望处相关':>11s}")
    for k in np.linspace(nk * 0.3, nk * 0.7, a.probes).astype(int):
        tk = thick[k].ravel()
        tk = (tk - tk.mean()) / (tk.std() + 1e-8)
        best, bj = -2.0, -1
        at_expect = None
        for j in range(0, D - r + 1):
            w = ((cs[j + r] - cs[j]) / r).ravel().astype(np.float32)
            sd = w.std()
            if sd < 1e-6: continue
            c = float(((w - w.mean()) / sd * tk).mean())
            if j == k * r: at_expect = c
            if c > best: best, bj = c, j
        print(f"{k:>8d}{k*r:>9d}{bj:>9d}{bj-k*r:>7d}{best:>9.3f}"
              f"{(at_expect if at_expect is not None else float('nan')):>11.3f}")
