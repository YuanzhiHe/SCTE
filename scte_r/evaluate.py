"""Evaluate a trained SCTE-R checkpoint: image quality + densitometry accuracy.

Reports, per case: PSNR/SSIM (image), LAA-950 / Perc15 error vs 1mm reference,
and the reliability flag. Aggregates CCC/Bland-Altman inputs for Fig.3/Fig.6.

Hyper-parameters (patch, slice width, model width) come from configs/default.yaml
so the evaluated model matches the trained one; same-named CLI flags override.
"""
import argparse
import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .model import build_model
from .datasets import (SyntheticPairedDataset, SimulatedThickDataset,
                       PreparedPairDataset, RealPairedDataset)
from . import metrics

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "default.yaml")


def build_dataset(args):
    if args.data == "synthetic":
        return SyntheticPairedDataset(n=args.n, downsample=args.downsample, seed=99)
    if args.data == "simulated":
        return SimulatedThickDataset(args.root, patch=tuple(args.patch),
                                     downsample=args.downsample,
                                     slice_fwhm_mm=args.slice_fwhm)
    if args.data == "pairs":
        return PreparedPairDataset(args.root, patch=tuple(args.patch),
                                   downsample=args.downsample)
    return RealPairedDataset(args.audit, args.root)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--root", default=None); ap.add_argument("--audit", default=None)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--downsample", type=int, default=None)
    ap.add_argument("--slice_fwhm", type=float, default=None)
    ap.add_argument("--patch", type=int, nargs=3, default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--arc", action="store_true", help="evaluate the ARC-CT variant")
    ap.add_argument("--op_source", default="protocol", choices=["protocol", "upsampled", "recon"])
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--no_identify", action="store_true")
    ap.add_argument("--csv", default=None, help="write per-case metrics here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}
    d, m = cfg.get("data", {}), cfg.get("model", {})
    if args.downsample is None: args.downsample = d.get("downsample", 5)
    if args.slice_fwhm is None: args.slice_fwhm = d.get("slice_fwhm_mm", 5.0)
    if args.patch is None:      args.patch = d.get("patch", [40, 64, 64])

    ds = build_dataset(args)
    dl = DataLoader(ds, batch_size=1)
    extra = {}
    if args.arc or args.iters > 1:
        from .agentic import TailCalibration
        extra = dict(arc=True, op_source=args.op_source,
                     identify=not args.no_identify,
                     calibration=TailCalibration.load(args.calibration))
    model = build_model(iters=args.iters, downsample=args.downsample,
                        slice_fwhm_mm=args.slice_fwhm,
                        base_ch=m.get("base_ch", 32), n_blocks=m.get("n_blocks", 6),
                        cond_dim=m.get("cond_dim", 32), dc_steps=m.get("dc_steps", 1),
                        n_kernels=m.get("n_kernels", 16), **extra).to(args.device).eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=args.device)
        if extra and not any(k.startswith("net.") for k in sd):
            sd = {f"net.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd, strict=not extra)

    laa_true, laa_pred, laa_thick, p15_true, p15_pred, p15_thick = [], [], [], [], [], []
    psnrs, psnrs_thick, ssims, rels = [], [], [], []
    for y, x, kid in dl:
        y, x, kid = y.to(args.device), x.to(args.device), kid.to(args.device)
        out = model(y, kid)
        xh = out["x_hat"]
        y_up = model.op.upsample_to_grid(y, xh.shape[2])
        laa_true.append(metrics.laa950(x).item()); laa_pred.append(metrics.laa950(xh).item())
        laa_thick.append(metrics.laa950(y_up).item())
        p15_true.append(metrics.perc15(x).item()); p15_pred.append(metrics.perc15(xh).item())
        p15_thick.append(metrics.perc15(y_up).item())
        psnrs.append(metrics.psnr(xh, x).item()); psnrs_thick.append(metrics.psnr(y_up, x).item())
        ssims.append(metrics.ssim(xh, x).item())
        rels.append(out["reliability"].mean().item())

    laa_true, laa_pred, laa_thick = map(np.array, (laa_true, laa_pred, laa_thick))
    p15_true, p15_pred, p15_thick = map(np.array, (p15_true, p15_pred, p15_thick))
    ccc_recon = metrics.ccc(torch.tensor(laa_pred), torch.tensor(laa_true)).item()
    ccc_thick = metrics.ccc(torch.tensor(laa_thick), torch.tensor(laa_true)).item()
    print(f"n={len(laa_true)}")
    print(f"PSNR         thick   : {np.mean(psnrs_thick):.2f} dB")
    print(f"PSNR                 : {np.mean(psnrs):.2f} dB")
    print(f"SSIM                 : {np.mean(ssims):.4f}")
    print(f"LAA-950 bias thick   : {np.mean(laa_thick-laa_true):+.2f} pp")
    print(f"LAA-950 bias SCTE-R  : {np.mean(laa_pred -laa_true):+.2f} pp")
    print(f"LAA-950 MAE  thick   : {np.mean(np.abs(laa_thick-laa_true)):.2f} pp")
    print(f"LAA-950 MAE  SCTE-R  : {np.mean(np.abs(laa_pred -laa_true)):.2f} pp")
    print(f"Perc15  MAE  thick   : {1000*np.mean(np.abs(p15_thick-p15_true)):.2f} HU")
    print(f"Perc15  MAE  SCTE-R  : {1000*np.mean(np.abs(p15_pred -p15_true)):.2f} HU")
    print(f"LAA-950 CCC (thick vs ref): {ccc_thick:.3f}")
    print(f"LAA-950 CCC (recon vs ref): {ccc_recon:.3f}")
    print(f"reliability (mean)   : {np.mean(rels):.3f}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["case", "laa_ref", "laa_thick", "laa_recon",
                        "perc15_ref_HU", "perc15_thick_HU", "perc15_recon_HU",
                        "psnr_thick", "psnr_recon", "ssim", "reliability"])
            names = [os.path.basename(p) for p in getattr(ds, "files", [])] or \
                    [str(i) for i in range(len(laa_true))]
            for i, nm in enumerate(names[:len(laa_true)]):
                w.writerow([nm, f"{laa_true[i]:.4f}", f"{laa_thick[i]:.4f}", f"{laa_pred[i]:.4f}",
                            f"{1000*p15_true[i]:.2f}", f"{1000*p15_thick[i]:.2f}",
                            f"{1000*p15_pred[i]:.2f}", f"{psnrs_thick[i]:.3f}",
                            f"{psnrs[i]:.3f}", f"{ssims[i]:.4f}", f"{rels[i]:.4f}"])
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
