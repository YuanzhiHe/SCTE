"""Fit the effective slice width of a REAL thick series from a real thin/thick pair.

The training operator assumes A = Gaussian(FWHM w0) along z + strided average.
On real data w0 is unknown, and if the true SSP is narrower/wider than the one the
model was trained with, the reconstruction inherits a systematic densitometry bias.
This is the measurement M1 is meant to make; here it is done in closed form by a
grid search, so the sim-to-real gap can be attributed (operator width vs noise).

  python scripts/fit_ssp.py --root DATA/public_pairs_test --downsample 5
"""
import argparse
import os

import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--grid", type=float, nargs=3, default=[2.0, 12.0, 0.25],
                    help="lo hi step of the FWHM grid (mm)")
    ap.add_argument("--slab", type=int, default=80, help="thin slices used per case")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    cases = sorted(f for f in os.listdir(args.root) if f.endswith("_thin.npy"))
    if args.n:
        cases = cases[:args.n]
    r = args.downsample
    grid = np.arange(*[args.grid[0], args.grid[1] + 1e-9, args.grid[2]])
    best_w, ratios = [], []

    for c in cases:
        thin = np.load(os.path.join(args.root, c), mmap_mode="r")
        thick = np.load(os.path.join(args.root, c.replace("_thin", "_thick")), mmap_mode="r")
        d = min(args.slab, thin.shape[0] // r * r)
        z0 = (thin.shape[0] - d) // 2; z0 -= z0 % r
        x = torch.from_numpy(np.ascontiguousarray(thin[z0:z0 + d])[None, None]) / 1000.0
        y = torch.from_numpy(np.ascontiguousarray(
            thick[z0 // r:z0 // r + d // r])[None, None]) / 1000.0
        errs = []
        for w in grid:
            op = SSPForwardOperator(slice_fwhm_mm=float(w), downsample=r)
            errs.append(float(((op(x) - y) ** 2).mean()))
        w_hat = float(grid[int(np.argmin(errs))])
        op = SSPForwardOperator(slice_fwhm_mm=w_hat, downsample=r)
        resid = (op(x) - y)
        # noise the operator cannot explain, relative to the thick slab's own std
        ratio = float(resid.std() / y.std())
        best_w.append(w_hat); ratios.append(ratio)
        print(f"{c.replace('_thin.npy',''):<14} w0_hat={w_hat:5.2f} mm  "
              f"resid/std={ratio:.3f}  mse_min={min(errs):.5f}")

    w = np.array(best_w)
    print(f"\nn={len(w)}  effective slice width: median {np.median(w):.2f} mm, "
          f"mean {w.mean():.2f} ± {w.std():.2f}, range [{w.min():.2f}, {w.max():.2f}]")
    print(f"unexplained residual / thick std: median {np.median(ratios):.3f}")
    print(f"(training used {args.downsample}.0-{args.downsample + 2}.0 mm jitter "
          f"around the nominal width)")


if __name__ == "__main__":
    main()
