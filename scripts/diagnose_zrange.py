"""Find where the thick series actually sits inside the thin one.

A relative residual near 1.0 means the operator's prediction is no better than the
series mean - the two volumes have no usable correspondence at the offsets searched so
far. Both prep_pairs (+-4 slabs) and realign_pairs (0..9 thin slices) look only within
about +-20 thin slices, so a pair whose scan ranges differ by more than that is invisible
to them and comes out as garbage rather than as an error.

This searches the FULL range and also tries a flipped z axis, which catches
feet-first/head-first storage differences.

  python scripts/diagnose_zrange.py --root PRIVATE/henan/pairs --n 12
"""
import argparse, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--n', type=int, default=12)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--slice_fwhm', type=float, default=6.25)
a = ap.parse_args()
r = a.downsample
op = SSPForwardOperator(slice_fwhm_mm=a.slice_fwhm, downsample=r)


def profile(vol):
    """One number per slice - enough to locate a series inside another, and cheap."""
    return vol.reshape(vol.shape[0], -1).mean(1)


def best_lag(pt, pk):
    """Cross-correlate the thin profile (decimated) against the thick one, all lags."""
    x = pt[: (len(pt) // r) * r].reshape(-1, r).mean(1)
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (pk - pk.mean()) / (pk.std() + 1e-8)
    if len(y) > len(x):
        return None, 0.0
    c = np.correlate(x, y, mode='valid') / len(y)
    j = int(np.argmax(c))
    return j, float(c[j])


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
print(f"{'case':<14s}{'thin层':>7s}{'thick层':>8s}{'比值':>7s}"
      f"{'最佳slab位移':>13s}{'相关':>7s}{'翻转相关':>9s}{'判定':>10s}")
bad = 0
for f in files:
    thin = np.load(os.path.join(a.root, f), mmap_mode='r')
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')), mmap_mode='r')
    pt, pk = profile(np.asarray(thin, np.float32)), profile(np.asarray(thick, np.float32))
    lag, c = best_lag(pt, pk)
    lagf, cf = best_lag(pt, pk[::-1])
    ratio = thin.shape[0] / max(thick.shape[0], 1)
    if cf > c + 0.15:   verdict = 'z 方向相反'
    elif c < 0.5:       verdict = '无对应'
    elif lag and lag * r > 20: verdict = f'超窗口({lag*r}层)'
    else:               verdict = 'OK'
    if verdict != 'OK': bad += 1
    print(f"{f[:-9][:14]:<14s}{thin.shape[0]:7d}{thick.shape[0]:8d}{ratio:7.2f}"
          f"{(lag if lag is not None else -1):13d}{c:7.3f}{cf:9.3f}{verdict:>10s}")
print(f"\n{bad}/{len(files)} 例有问题")
print("\nslab 位移 = 厚层第 0 层对应薄层的第 (位移×5) 层。prep_pairs 只搜 ±4，")
print("realign_pairs 只搜 0-9 薄层 —— 位移超过这个范围时两者都找不到，")
print("而且不会报错，只会产出无意义的配对。")
