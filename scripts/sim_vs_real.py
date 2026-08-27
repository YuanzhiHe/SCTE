"""Sim-vs-real check: run one SCTE-R checkpoint on a SIMULATED thick slab and on
the REAL thick series of the same case, and compare the densitometry error.

This is the cheapest available proxy for "will the public-pretrained model hold up
on a real scanner's thick series?" — the question that decides whether stage 2 can
start from `public.pt` directly or needs domain adaptation first. RPLHR-CT ships
real 1mm/5mm pairs, so no private data is needed to run it.

Usage
-----
  python scripts/sim_vs_real.py --thin DATA/rplhr/val/1mm --thick DATA/rplhr/val/5mm \
         --ckpt public.pt --rescale_hu 3072 -1024 --csv results/sim_vs_real.csv

Notes
-----
* Volumes are matched by file name. z-alignment between the two series is not
  guaranteed, so the slab offset is estimated per case by minimising
  ||real_thick - A(thin)|| over a small range of shifts.
* The patch is the most lung-filled (40,64,64) box of the thin volume, with its z
  origin snapped to a multiple of the downsample factor so the slabs line up.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import SimpleITK as sitk
from prep_public import lung_bbox
from scte_r.model import build_model
from scte_r.forward_operator import SSPForwardOperator
from scte_r.datasets import HU_SCALE
from scte_r import metrics

HU_CLIP = (-1000.0, 200.0)


def read_hu(path, rescale):
    a = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
    if rescale:
        a = a * rescale[0] + rescale[1]
    return np.clip(a, *HU_CLIP)


def lung_origin(thin, patch, stride=8):
    """Most lung-filled patch origin in the thin volume."""
    v = thin[::stride, ::stride, ::stride]
    lung = ((v > -990.0) & (v < -500.0)).astype(np.float32)
    box = tuple(max(1, p // stride) for p in patch)
    score = ndimage.uniform_filter(lung, size=box, mode="constant")
    c = np.unravel_index(score.argmax(), score.shape)
    start = [int((ci - bi // 2) * stride) for ci, bi in zip(c, box)]
    return [int(np.clip(s, 0, max(0, d - p))) for s, d, p in zip(start, thin.shape, patch)]


def best_z_offset(thin, thick, op, r, search=4):
    """Slab offset (in thick slices) that best matches the real thick to A(thin)."""
    x = torch.from_numpy(thin[None, None]) / HU_SCALE
    sim = op(x)[0, 0].numpy()                      # (D//r, H, W)
    n = min(sim.shape[0], thick.shape[0])
    best, best_err = 0, None
    for off in range(-search, search + 1):
        a0, b0 = max(0, off), max(0, -off)
        m = min(n - a0, n - b0)
        if m < 4:
            continue
        err = np.mean((sim[a0:a0 + m] - thick[b0:b0 + m] / HU_SCALE) ** 2)
        if best_err is None or err < best_err:
            best, best_err = off, err
    return best, float(best_err)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", required=True); ap.add_argument("--thick", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rescale_hu", type=float, nargs=2, default=None)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--slice_fwhm", type=float, default=5.0)
    ap.add_argument("--patch", type=int, nargs=3, default=[40, 64, 64])
    ap.add_argument("--thin_noise", type=float, default=15.0)
    ap.add_argument("--thick_noise", type=float, default=7.0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    names = sorted(f for f in os.listdir(args.thin) if f.endswith((".nii", ".nii.gz")))
    if args.n:
        names = names[:args.n]
    model = build_model(iters=1, downsample=args.downsample,
                        slice_fwhm_mm=args.slice_fwhm).to(args.device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    op = SSPForwardOperator(slice_fwhm_mm=args.slice_fwhm, downsample=args.downsample)
    pd, ph, pw = args.patch
    r = args.downsample
    rows, offs = [], []

    for k, nm in enumerate(names):
        tp = os.path.join(args.thick, nm)
        if not os.path.exists(tp):
            print(f"[skip] {nm}: no thick counterpart"); continue
        thin = read_hu(os.path.join(args.thin, nm), args.rescale_hu)
        thick = read_hu(tp, args.rescale_hu)
        # select the patch exactly like training does: inside the lung bounding
        # box of the thin volume, at its most lung-filled position
        bb = lung_bbox(thin)
        z0, y0, x0 = (o + s.start for o, s in zip(lung_origin(thin[bb], args.patch), bb))
        z0 -= z0 % r                                   # snap so slabs line up
        off, _ = best_z_offset(thin[:, y0:y0 + ph, x0:x0 + pw],
                               thick[:, y0:y0 + ph, x0:x0 + pw], op, r)
        offs.append(off)
        zt = z0 // r + off
        if zt < 0 or zt + pd // r > thick.shape[0] or z0 + pd > thin.shape[0]:
            print(f"[skip] {nm}: slab out of range"); continue

        x = torch.from_numpy(thin[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw][None, None]) / HU_SCALE
        y_real = torch.from_numpy(
            thick[zt:zt + pd // r, y0:y0 + ph, x0:x0 + pw][None, None]) / HU_SCALE
        g = torch.Generator().manual_seed(k)
        xn = x + torch.randn(x.shape, generator=g) * (args.thin_noise / HU_SCALE)
        y_sim = op(xn) + torch.randn(y_real.shape, generator=g) * (args.thick_noise / HU_SCALE)

        x = x.to(args.device)
        out = {}
        for tag, y in (("sim", y_sim.to(args.device)), ("real", y_real.to(args.device))):
            o = model(y)
            out[tag] = dict(
                laa_rec=metrics.laa950(o["x_hat"]).item(),
                laa_thick=metrics.laa950(model.op.upsample_to_grid(y, pd)).item(),
                p15_rec=1000 * metrics.perc15(o["x_hat"]).item(),
                p15_thick=1000 * metrics.perc15(model.op.upsample_to_grid(y, pd)).item(),
                psnr=metrics.psnr(o["x_hat"], x).item(),
                rel=o["reliability"].mean().item())
        rows.append(dict(case=nm, off=off, laa_ref=metrics.laa950(x).item(),
                         p15_ref=1000 * metrics.perc15(x).item(), **{
                             f"{t}_{k2}": v for t, d in out.items() for k2, v in d.items()}))
        print(f"[{len(rows):3d}] {nm} off={off:+d} LAA ref={rows[-1]['laa_ref']:.2f} "
              f"sim(thick={out['sim']['laa_thick']:.2f} rec={out['sim']['laa_rec']:.2f}) "
              f"real(thick={out['real']['laa_thick']:.2f} rec={out['real']['laa_rec']:.2f})")

    if not rows:
        sys.exit("no usable pairs")
    A = {k: np.array([r_[k] for r_ in rows], dtype=float) for k in rows[0] if k != "case"}
    ref, p15ref = A["laa_ref"], A["p15_ref"]
    print(f"\nn={len(rows)}   z-offset used: "
          f"{dict(zip(*np.unique(offs, return_counts=True)))}")
    print(f"{'':22}{'LAA MAE':>9}{'LAA med':>9}{'LAA bias':>10}{'LAA CCC':>9}"
          f"{'P15 MAE':>9}{'PSNR':>8}")
    for tag in ("sim", "real"):
        for what in ("thick", "rec"):
            v, p = A[f"{tag}_laa_{what}"], A[f"{tag}_p15_{what}"]
            ccc = metrics.ccc(torch.tensor(v), torch.tensor(ref)).item()
            ps = np.mean(A[f"{tag}_psnr"]) if what == "rec" else float("nan")
            label = f"{tag} {'thick input' if what == 'thick' else 'SCTE-R'}"
            print(f"{label:22}{np.mean(np.abs(v-ref)):9.2f}{np.median(np.abs(v-ref)):9.2f}"
                  f"{np.mean(v-ref):10.2f}{ccc:9.3f}{np.mean(np.abs(p-p15ref)):9.2f}{ps:8.2f}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r_ in rows:
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in r_.items()})
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
