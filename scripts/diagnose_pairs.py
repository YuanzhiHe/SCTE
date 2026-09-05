"""Why does the forward operator not explain this cohort?

A relative residual near 0.75 means the slab-average model accounts for only about a
quarter of the difference between the two series. Re-aligning barely moves it (6% here
versus 55% on a cohort with a genuine sub-slab offset), so the offset search is finding
minima in noise - which is also why the chosen shifts scatter instead of agreeing.

The usual cause in hospital archives is that the two series were reconstructed with
different kernels: a sharp lung kernel for the 1 mm and a smooth standard kernel for the
5 mm. That is not a z problem at all, and no realignment fixes it.

This separates the possibilities:
  * kernel / in-plane mismatch -> the residual collapses once both series are blurred
    in-plane, because the disagreement lives in high spatial frequencies
  * genuine z problem          -> in-plane blurring changes little
  * not the same acquisition   -> nothing helps, and the residual stays high

  python scripts/diagnose_pairs.py --root PRIVATE/henan/pairs --n 12
"""
import argparse, csv, os, sys
import numpy as np, torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--n', type=int, default=12)
ap.add_argument('--downsample', type=int, default=5)
a = ap.parse_args()
r = a.downsample


def resid(thin, thick, w, blur_xy=0):
    op = SSPForwardOperator(slice_fwhm_mm=w, downsample=r)
    x = torch.from_numpy(thin[None, None] / 1000.).float()
    y = torch.from_numpy(thick[None, None] / 1000.).float()
    if blur_xy:
        k = 2 * blur_xy + 1
        g = torch.exp(-(torch.arange(k).float() - blur_xy) ** 2 / (2 * (blur_xy / 2 + .5) ** 2))
        g = (g / g.sum()).view(1, 1, 1, 1, k)
        for d in (3, 4):
            gg = g.transpose(4, d)
            x = F.conv3d(x, gg, padding=[0, 0, 0][:2] + [0] if False else
                         tuple(0 if i != d - 2 else blur_xy for i in range(3)))
            y = F.conv3d(y, gg, padding=tuple(0 if i != d - 2 else blur_xy for i in range(3)))
    ax = op(x)
    n = min(ax.shape[2], y.shape[2])
    ax, y = ax[:, :, :n], y[:, :, :n]
    d = float((ax - y).pow(2).mean().sqrt())
    s = float((y - y.mean()).pow(2).mean().sqrt())
    return d / max(s, 1e-8)


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
print(f"{'case':<14s}{'w=5':>8s}{'w=6.25':>8s}{'w=8':>8s}{'w=12':>8s}"
      f"{'面内模糊2':>11s}{'面内模糊4':>11s}")
R = {k: [] for k in ('w5', 'w625', 'w8', 'w12', 'b2', 'b4')}
for f in files:
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    c = thin.shape[1] // 4
    thin = thin[:, c:3 * c, c:3 * c]; thick = thick[:, c:3 * c, c:3 * c]   # centre crop
    v = [resid(thin, thick, w) for w in (5., 6.25, 8., 12.)]
    b = [resid(thin, thick, 6.25, blur_xy=k) for k in (2, 4)]
    for k, x in zip(R, v + b): R[k].append(x)
    print(f"{f[:-9][:14]:<14s}" + ''.join(f"{x:8.3f}" for x in v) + ''.join(f"{x:11.3f}" for x in b))

m = {k: float(np.mean(v)) for k, v in R.items()}
print(f"\n{'均值':<14s}" + ''.join(f"{m[k]:8.3f}" for k in ('w5','w625','w8','w12'))
      + ''.join(f"{m[k]:11.3f}" for k in ('b2','b4')))
best_w = min(('w5','w625','w8','w12'), key=lambda k: m[k])
print(f"\n最佳层厚 {dict(w5='5',w625='6.25',w8='8',w12='12')[best_w]} mm，残差 {m[best_w]:.3f}")
drop = 1 - m['b4'] / m['w625']
print(f"面内模糊 4 后残差 {m['w625']:.3f} -> {m['b4']:.3f}（降 {100*drop:.0f}%）")
print()
if drop > 0.4:
    print("→ 判定：**重建核不匹配**。分歧集中在高空间频率，与 z 无关。")
    print("  薄层多半是肺算法(锐)、厚层是标准算法(平滑)。重切对齐不会有帮助。")
elif m[best_w] < 0.3:
    print("→ 判定：算子宽度设错。用上面那个最佳宽度重跑即可。")
else:
    print("→ 判定：面内模糊无改善且各宽度都差 —— 两个序列可能不是同一次采集，")
    print("  或存在 FOV/重建中心差异。查 pair_audit.csv 的 off/sub 列与协议元数据。")
