"""Classical, non-learned baseline: iterative through-plane deconvolution.

Why this arm exists. Every model we certified is OURS, so "the objective buys its
densitometric accuracy with a global attenuation shift" could in principle be a
property of this codebase rather than of the task. A Landweber deconvolution with
the protocol slice-sensitivity profile has no free parameters that could learn a
shift: it descends ||A_w x - y||^2 from the up-sampled thick volume. If it shows a
small delta_hat (no displacement) and no densitometric gain, then displacement is
what the LEARNED objective adds — which is the claim.

  python scripts/baseline_deconv.py --root DATA/public_pairs_test \
         --calibration runs/protocol_rplhr_5mm.json --iters 30 --step 1.0
"""
import argparse, csv, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r import metrics
from scte_r.agentic import Arbiter, DensitometrySolver, TailCalibration
from scte_r.datasets import PreparedPairDataset
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--calibration", required=True)
ap.add_argument("--downsample", type=int, default=5)
ap.add_argument("--patch", type=int, nargs=3, default=[40, 64, 64])
ap.add_argument("--iters", type=int, default=30, help="Landweber iterations")
ap.add_argument("--step", type=float, default=1.0, help="Landweber step size")
ap.add_argument("--lung_mask", action="store_true",
                help="use the TotalSegmentator lobe mask for the densitometry denominator")
ap.add_argument("--csv", default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

pj = json.load(open(a.calibration))
w = float(pj.get("protocol_w", 5.0))
cal = TailCalibration.load(a.calibration)
dens = DensitometrySolver(a.downsample, calibration=cal).to(a.device)
arb = Arbiter(float(pj.get("tau_delta", 2.0)), float(pj.get("tau_rho", 0.15)),
              float(pj.get("tau_s", 2.0)))
op = SSPForwardOperator(slice_fwhm_mm=w, downsample=a.downsample)
ds = PreparedPairDataset(a.root, patch=tuple(a.patch), downsample=a.downsample)
print(f"Landweber deconvolution: w={w} mm, {a.iters} iters, step {a.step}")

rows = []
with torch.no_grad():
    for i in range(len(ds)):
        y, x, _ = ds[i]
        y = y[None].to(a.device); x = x[None].to(a.device)
        lung = None
        if a.lung_mask:
            mp = ds.files[i].replace("_thin.npy", "_lung.npy")
            if os.path.exists(mp):
                g = ds._crop_origin(np.load(mp, mmap_mode="r").shape, i,
                                    np.random.default_rng(i + 12345))
                pd_, ph_, pw_ = ds.patch
                lab = np.load(mp, mmap_mode="r")[g[0]:g[0]+pd_, g[1]:g[1]+ph_, g[2]:g[2]+pw_]
                lung = torch.from_numpy(np.ascontiguousarray(lab))[None, None].to(a.device) > 0
        xh = op.upsample_to_grid(y, x.shape[2]).clone()
        for _ in range(a.iters):                       # x <- x - step * A^T(Ax - y)
            xh = xh - a.step * op.upsample_to_grid(op(xh) - y, x.shape[2])
        c = dens(xh, y, torch.full((1,), w, device=a.device))
        v = arb(c)[0]
        rows.append(dict(case=os.path.basename(ds.files[i]),
                         delta_HU=1000 * float(c["delta_hat"][0]),
                         rho_struct=float(c["rho_struct"][0]),
                         s=("" if c["s"][0] != c["s"][0] else round(float(c["s"][0]), 3)),
                         verdict=v[0], violated=v[1] or "",
                         laa_ref=metrics.laa950(x, lung).item(),
                         laa_rec=metrics.laa950(xh, lung).item(),
                         p15_ref=1000 * metrics.perc15(x, lung).item(),
                         p15_rec=1000 * metrics.perc15(xh, lung).item(),
                         psnr=metrics.psnr(xh, x).item()))

A = {k: np.array([r[k] for r in rows], dtype=float)
     for k in ("delta_HU", "rho_struct", "laa_ref", "laa_rec", "p15_ref", "p15_rec", "psnr")}
print(f"n={len(rows)}")
print(f"LAA-950 MAE  : {np.mean(np.abs(A['laa_rec']-A['laa_ref'])):.2f} pp")
print(f"LAA-950 CCC  : {metrics.ccc(torch.tensor(A['laa_rec']), torch.tensor(A['laa_ref'])).item():.3f}")
print(f"Perc15  MAE  : {np.mean(np.abs(A['p15_rec']-A['p15_ref'])):.2f} HU")
print(f"PSNR         : {np.mean(A['psnr']):.2f} dB")
print(f"delta_hat    : {A['delta_HU'].mean():+.2f} +- {A['delta_HU'].std():.2f} HU")
print(f"certified    : {sum(r['verdict']=='certified' for r in rows)}/{len(rows)}")
import collections; print("flagged by  :", dict(collections.Counter(r['violated'] or r['verdict'] for r in rows)))
if a.csv:
    with open(a.csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
    print("wrote", a.csv)
