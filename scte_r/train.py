"""Training loop for SCTE-R. Works on synthetic / public-simulated / real data.

Hyper-parameters come from configs/default.yaml; same-named CLI flags override.

Examples:
  # smoke (no data):
  python -m scte_r.train --data synthetic --epochs 1 --steps 5
  # public pretraining (thin .npy volumes under DATA/):
  python -m scte_r.train --data simulated --root DATA/public_thin --epochs 50
  # private fine-tune / validate:
  python -m scte_r.train --data real --audit pair_audit.csv --root /path/cases --epochs 20
"""
import argparse
import os
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from .model import build_model
from .losses import SCTERLoss
from .datasets import (SyntheticPairedDataset, SimulatedThickDataset,
                       PreparedPairDataset, RealPairedDataset)
from . import metrics

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "default.yaml")


def build_dataset(args, root=None, random_crop=True, seed=0):
    if args.data == "synthetic":
        return SyntheticPairedDataset(n=args.n, downsample=args.downsample)
    if args.data == "simulated":
        return SimulatedThickDataset(root or args.root, patch=tuple(args.patch),
                                     downsample=args.downsample,
                                     slice_fwhm_mm=args.slice_fwhm,
                                     random_crop=random_crop, seed=seed)
    if args.data == "pairs":
        ds = PreparedPairDataset(root or args.root, patch=tuple(args.patch),
                                 downsample=args.downsample,
                                 random_crop=random_crop, seed=seed)
        ds.with_base = bool(getattr(args, "base_recon", False))
        return ds
    if args.data == "real":
        return RealPairedDataset(args.audit, root or args.root, patch=tuple(args.patch))
    raise ValueError(args.data)


@torch.no_grad()
def validate(model, dl, device):
    """Held-out densitometry check: LAA-950 MAE of the thick input vs the recon."""
    model.eval()
    laa_t, laa_p, laa_y, psnrs = [], [], [], []
    for y, x, kid in dl:
        y, x, kid = y.to(device), x.to(device), kid.to(device)
        out = model(y, kid)
        xh = out["x_hat"]
        laa_t.append(metrics.laa950(x).item())
        laa_p.append(metrics.laa950(xh).item())
        laa_y.append(metrics.laa950(model.op.upsample_to_grid(y, xh.shape[2])).item())
        psnrs.append(metrics.psnr(xh, x).item())
    model.train()
    laa_t, laa_p, laa_y = map(np.array, (laa_t, laa_p, laa_y))
    return dict(mae_recon=float(np.mean(np.abs(laa_p - laa_t))),
                mae_thick=float(np.mean(np.abs(laa_y - laa_t))),
                psnr=float(np.mean(psnrs)),
                ccc=float(metrics.ccc(torch.tensor(laa_p), torch.tensor(laa_t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "simulated", "pairs", "real"],
                    help="simulated = thick synthesised by the SSP operator; "
                         "pairs = real thin/thick pairs from scripts/prep_pairs.py")
    ap.add_argument("--root", default=None)
    ap.add_argument("--val_root", default=None, help="held-out thin .npy dir")
    ap.add_argument("--audit", default=None)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--downsample", type=int, default=None)
    ap.add_argument("--slice_fwhm", type=float, default=None)
    ap.add_argument("--patch", type=int, nargs=3, default=None)
    ap.add_argument("--iters", type=int, default=1,
                    help="1 = single-pass SCTE_R; >1 = iterative/agentic refinement")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=0, help="cap steps/epoch (0=all)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--w_dc", type=float, default=None)
    ap.add_argument("--w_traj", type=float, default=None)
    ap.add_argument("--w_nps", type=float, default=None)
    ap.add_argument("--arc", action="store_true",
                    help="ARC-CT: identify the acquisition per scan and emit a certificate")
    ap.add_argument("--w_delta", type=float, default=0.0,
                    help="weight of the displacement penalty |delta_hat| (ARC-CT only)")
    ap.add_argument("--op_source", default="protocol", choices=["protocol", "upsampled", "recon"],
                    help="what the operator fit runs against (see agentic.OperatorSolver)")
    ap.add_argument("--calibration", default=None,
                    help="tail-recovery calibration json (scripts/fit_tail_calibration.py)")
    ap.add_argument("--no_identify", action="store_true",
                    help="ablation (a): freeze the width at the nominal value")
    ap.add_argument("--e_mask", type=int, default=8, help="lung-mask erosion radius")
    ap.add_argument("--base_ch", type=int, default=None, help="override model width")
    ap.add_argument("--n_blocks", type=int, default=None, help="override model depth")
    ap.add_argument("--base_recon", action="store_true",
                    help="condition the flow on the cached <case>_base.npy strong "
                         "reconstruction instead of plain up-sampling")
    ap.add_argument("--flow", action="store_true",
                    help="flow-matching decoder: SAMPLE the thin volume instead of "
                         "regressing it (scte_r/flow.py). The objective is the "
                         "conditional flow-matching loss; the certificate is measured, "
                         "never trained on.")
    ap.add_argument("--residual_scale", type=float, default=0.05,
                    help="kHU scale of the residual the flow transports (0.05 = 50 HU)")
    ap.add_argument("--flow_steps", type=int, default=16, help="Euler steps at sampling")
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="measurement-guidance step applied at EVERY sampling step")
    ap.add_argument("--w_bio", type=float, default=0.0,
                    help="published-style biomarker multi-task loss: match LAA-950/910 "
                         "and Perc15/10/5 directly at native resolution")
    ap.add_argument("--w_quant", type=float, default=0.0,
                    help="weight of the lung-histogram quantile-matching loss (HU)")
    ap.add_argument("--clip", type=float, default=0.0, help="grad-norm clip (0=off)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA decay for the saved weights (0=off, e.g. 0.999). The "
                         "trajectory guardrail makes single-step weights oscillate; "
                         "the EMA of them is what gets validated and checkpointed.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt", default="scte_r.pt")
    ap.add_argument("--init", default=None, help="warm-start checkpoint (public->private)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}
    d, m, l, t = (cfg.get(k, {}) for k in ("data", "model", "loss", "train"))
    if args.downsample is None: args.downsample = d.get("downsample", 5)
    if args.slice_fwhm is None: args.slice_fwhm = d.get("slice_fwhm_mm", 5.0)
    if args.patch is None:      args.patch = d.get("patch", [40, 64, 64])
    if args.epochs is None:     args.epochs = t.get("epochs", 1)
    if args.bs is None:         args.bs = t.get("batch_size", 2)
    if args.lr is None:         args.lr = t.get("lr", 1e-3)
    if args.w_dc is None:       args.w_dc = l.get("w_dc", 1.0)
    if args.w_traj is None:     args.w_traj = l.get("w_traj", 0.5)
    if args.w_nps is None:      args.w_nps = l.get("w_nps", 0.1)
    ladder = tuple(l.get("ladder", (0.0, 1.0, 2.0, 3.0)))
    if args.base_ch is not None:  m["base_ch"] = args.base_ch
    if args.n_blocks is not None: m["n_blocks"] = args.n_blocks

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = build_dataset(args, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=len(ds) > args.bs, persistent_workers=args.workers > 0)
    val_dl = None
    if args.val_root:
        val_ds = build_dataset(args, root=args.val_root, random_crop=False)
        val_dl = DataLoader(val_ds, batch_size=1, num_workers=args.workers)

    extra = {}
    if args.flow:
        from .agentic import TailCalibration
        cal = TailCalibration.load(args.calibration)
        extra = dict(flow=True, op_source=args.op_source, e_mask=args.e_mask,
                     identify=not args.no_identify, calibration=cal,
                     residual_scale=args.residual_scale, steps=args.flow_steps,
                     guidance=args.guidance)
    elif args.arc or args.iters > 1:
        from .agentic import TailCalibration
        extra = dict(arc=True, op_source=args.op_source, e_mask=args.e_mask,
                     identify=not args.no_identify,
                     calibration=TailCalibration.load(args.calibration))
    model = build_model(iters=args.iters, downsample=args.downsample,
                        slice_fwhm_mm=args.slice_fwhm,
                        base_ch=m.get("base_ch", 32), n_blocks=m.get("n_blocks", 6),
                        cond_dim=m.get("cond_dim", 32), dc_steps=m.get("dc_steps", 1),
                        n_kernels=m.get("n_kernels", 16), **extra).to(args.device)
    if args.init:
        sd = torch.load(args.init, map_location=args.device)
        if extra and not any(k.startswith("net.") for k in sd):
            sd = {f"net.{k}": v for k, v in sd.items()}   # single-pass ckpt -> ARC-CT
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-started from {args.init} "
              f"({len(missing)} missing / {len(unexpected)} unexpected keys)")
    loss_fn = SCTERLoss(model.op, w_dc=args.w_dc, w_traj=args.w_traj,
                        w_nps=args.w_nps, ladder=ladder, w_delta=args.w_delta,
                        w_quant=args.w_quant, w_bio=args.w_bio)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"cases={len(ds)} patch={tuple(args.patch)} downsample={args.downsample} "
          f"lr={args.lr} w_dc={args.w_dc} w_traj={args.w_traj} w_nps={args.w_nps} "
          f"w_delta={args.w_delta} w_quant={args.w_quant} arc={bool(extra)} identify={not args.no_identify} "
          f"clip={args.clip} device={args.device}")

    ema = ({k: v.detach().clone().float() for k, v in model.state_dict().items()}
           if args.ema else None)

    def swap_in(state):
        """Load `state`, return the weights it replaced (so they can be restored)."""
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        return cur

    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        run = {}
        for it, batch in enumerate(dl):
            if args.steps and it >= args.steps:
                break
            y, x, kid = (t.to(args.device) for t in batch[:3])
            base = batch[3].to(args.device) if len(batch) > 3 else None
            if args.flow:
                loss = model.flow_loss(y, x, kid, base=base)
                logs = {"flow": loss.item(), "total": loss.item()}
            else:
                out = model(y, kid)
                loss, logs = loss_fn(out, x, y)
            opt.zero_grad(); loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    msd = model.state_dict()
                    for k, v in ema.items():
                        if v.is_floating_point():
                            v.mul_(args.ema).add_(msd[k].float(), alpha=1 - args.ema)
                        else:
                            v.copy_(msd[k])
            for k, v in logs.items():
                run[k] = run.get(k, 0.0) + v
            if it % 10 == 0 and args.flow:
                print(f"ep{ep} it{it} flow={logs['flow']:.4f}")
            elif it % 10 == 0:
                laa_err = (metrics.laa950(out["x_hat"]) - metrics.laa950(x)).abs().item()
                print(f"ep{ep} it{it} loss={logs['total']:.3f} "
                      f"vox={logs['voxel']:.3f} dc={logs['dc']:.1f} "
                      f"traj={logs['traj']:.3f} nps={logs['nps']:.3f} |LAAerr|={laa_err:.2f}pp")
        nb = max(1, it + 1 if not args.steps else min(it + 1, args.steps))
        print(f"[epoch {ep}] mean " +
              " ".join(f"{k}={v/nb:.3f}" for k, v in run.items()))
        live = swap_in(ema) if ema is not None else None      # validate/save the EMA
        torch.save(model.state_dict(), args.ckpt)
        if val_dl is not None:
            v = validate(model, val_dl, args.device)
            flag = ""
            if v["mae_recon"] < best:
                best = v["mae_recon"]
                torch.save(model.state_dict(), args.ckpt.replace(".pt", "_best.pt"))
                flag = "  <- best"
            print(f"[val {ep}] PSNR={v['psnr']:.2f}dB  LAA MAE thick={v['mae_thick']:.2f}pp "
                  f"recon={v['mae_recon']:.2f}pp  CCC={v['ccc']:.3f}{flag}")
        if live is not None:
            model.load_state_dict(live)                       # back to the live weights
    print("saved", args.ckpt)


if __name__ == "__main__":
    main()
