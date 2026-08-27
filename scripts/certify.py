"""Run a checkpoint (or the perfect-reconstruction oracle) over prepared pairs and
write the ARC-CT certificate NEXT TO the densitometric endpoint, per case.

The harness refuses to print an endpoint without its certificate: that coupling is
the point of the method — a densitometric value from a restored thick-slice volume
is only interpretable together with the evidence that supports it.

  # estimator validation (S6): the certificate must return its null values when the
  # "reconstruction" IS the reference
  python scripts/certify.py --root DATA/public_pairs_test --oracle \
         --calibration runs/protocol_rplhr_5mm.json --csv results/cert_oracle.csv

  # a model
  python scripts/certify.py --root DATA/public_pairs_test --ckpt public.pt --arc \
         --calibration runs/protocol_rplhr_5mm.json --csv results/cert_public.csv

  # the bias-only control arm (a single scalar, fitted on --bias_pairs cases)
  python scripts/certify.py --root DATA/public_pairs_test --ckpt public.pt --arc \
         --bias_from DATA/public_pairs --bias_pairs 5 --calibration ... --csv ...
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r import metrics
from scte_r.agentic import Arbiter, TailCalibration
from scte_r.datasets import PreparedPairDataset, SimulatedThickDataset
from scte_r.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--simulated", action="store_true",
                    help="thin .npy dir with the thick slab SIMULATED by the SSP "
                         "operator (the 8 mm arm is simulated-only: no public real "
                         "8 mm pairs exist)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--oracle", action="store_true",
                    help="use the TRUE thin volume as the reconstruction (S6)")
    ap.add_argument("--arc", action="store_true", default=True)
    ap.add_argument("--base_ch", type=int, default=None, help="override model width")
    ap.add_argument("--n_blocks", type=int, default=None, help="override model depth")
    ap.add_argument("--flow", action="store_true", help="evaluate the flow decoder")
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="measurement-guidance step applied at EVERY sampling step")
    ap.add_argument("--samples", type=int, default=1,
                    help="flow decoder only: draw K samples and average the STATISTICS "
                         "(LAA, Perc15, certificate terms), never the images — LAA is a "
                         "functional, so its posterior mean is the right estimator, "
                         "whereas averaging the volumes would re-smooth the texture the "
                         "sampler exists to synthesise")
    ap.add_argument("--residual_scale", type=float, default=0.05)
    ap.add_argument("--dc_steps", type=int, default=None,
                    help="data-consistency projection steps applied to the sample")
    ap.add_argument("--dc_lr", type=float, default=None,
                    help="step size of that projection (default 0.5)")
    ap.add_argument("--no_identify", action="store_true",
                    help="ablation (a): nominal width instead of the protocol fit")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--base_recon", action="store_true",
                    help="feed the cached <case>_base.npy strong reconstruction to the "
                         "flow decoder as its base predictor (matches infer_volume)")
    ap.add_argument("--learned_op", default=None,
                    help="fitted forward operator (scripts/fit_forward_operator.py)")
    ap.add_argument("--bias_from", default=None,
                    help="fit an output-bias correction on pairs from this dir")
    ap.add_argument("--bias_pairs", type=int, default=5)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--slice_fwhm", type=float, default=5.0)
    ap.add_argument("--patch", type=int, nargs=3, default=[40, 64, 64])
    # defaults come from the protocol json when it carries them: they are the
    # |mean| + 2 sd of the certificate's NULL distribution, measured with a perfect
    # reconstruction on TRAINING pairs only (never on the evaluation set)
    ap.add_argument("--tau_delta", type=float, default=None)
    ap.add_argument("--tau_rho", type=float, default=None)
    ap.add_argument("--tau_s", type=float, default=None)
    ap.add_argument("--lung_mask", action="store_true",
                    help="use the TotalSegmentator lobe mask (<case>_lung.npy) for the "
                         "densitometry denominator and for the certificate's averaging "
                         "set, instead of the HU window")
    ap.add_argument("--per_lobe", action="store_true", help="also report per-lobe LAA")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cal = TailCalibration.load(args.calibration)
    pj = {}
    if args.calibration and os.path.exists(args.calibration):
        import json as _json
        pj = _json.load(open(args.calibration))
    for k, fallback in (("tau_delta", 2.0), ("tau_rho", 0.15), ("tau_s", 2.0)):
        if getattr(args, k) is None:
            setattr(args, k, float(pj.get(k, fallback)))
    print(f"thresholds: tau_delta={args.tau_delta} HU  tau_rho={args.tau_rho}  "
          f"tau_s={args.tau_s} sigma_v   (protocol: {pj.get('protocol', 'n/a')}, "
          f"w={pj.get('protocol_w', args.slice_fwhm)} mm)")
    ds = (SimulatedThickDataset(args.root, patch=tuple(args.patch),
                                downsample=args.downsample,
                                slice_fwhm_mm=args.slice_fwhm)
          if args.simulated else
          PreparedPairDataset(args.root, patch=tuple(args.patch),
                              downsample=args.downsample))
    if args.base_recon and not args.simulated:
        ds.with_base = True
    lop = None
    if args.learned_op:
        from scte_r.forward_learned import LearnedOperator
        d = torch.load(args.learned_op, map_location=args.device, weights_only=False)
        lop = LearnedOperator(d["downsample"], d["widths"], correction=d["correction"])
        lop.load_state_dict(d["state_dict"]); lop = lop.to(args.device).eval()
        for q in lop.parameters(): q.requires_grad_(False)
        print("learned operator: eff.FWHM %.2f mm, unexplained %.4f -> %.4f"
              % (lop.effective_fwhm(), d["unexplained_before"], d["unexplained_after"]))
    mkw = dict(iters=1, downsample=args.downsample, slice_fwhm_mm=args.slice_fwhm,
               calibration=cal, identify=not args.no_identify, learned_op=lop)
    if args.base_ch is not None:  mkw["base_ch"] = args.base_ch
    if args.n_blocks is not None: mkw["n_blocks"] = args.n_blocks
    if args.flow:
        mkw.update(flow=True, residual_scale=args.residual_scale, steps=args.flow_steps,
                   guidance=args.guidance)
    else:
        mkw.update(arc=True)
    if args.dc_steps is not None: mkw["dc_steps"] = args.dc_steps
    model = build_model(**mkw).to(args.device).eval()
    if args.dc_lr is not None:
        (model.net if hasattr(model, "net") else model).m3.lr = args.dc_lr
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=args.device)
        if not any(k.startswith("net.") for k in sd):
            sd = {f"net.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    bias = 0.0
    if args.bias_from:
        # The control arm the paper's claim is about: ONE scalar, fitted to minimise
        # the DENSITOMETRIC error on a handful of real pairs (not to match the mean -
        # mean-matching and LAA-matching pull in opposite directions here, which is
        # itself the point: the endpoint can be bought without fixing the histogram).
        src = (SimulatedThickDataset(args.bias_from, patch=tuple(args.patch),
                                     downsample=args.downsample,
                                     slice_fwhm_mm=args.slice_fwhm)
               if args.simulated else
               PreparedPairDataset(args.bias_from, patch=tuple(args.patch),
                                   downsample=args.downsample))
        cache = []
        with torch.no_grad():
            for i in range(min(args.bias_pairs, len(src))):
                y, x, _ = src[i]
                o = model(y[None].to(args.device))
                cache.append((o["x_hat"].detach(), x[None].to(args.device)))
        grid = np.arange(-0.030, 0.0301, 0.0005)          # kHU, i.e. -30..+30 HU
        errs = [float(np.mean([abs(metrics.laa950(xh + b).item() - metrics.laa950(xr).item())
                               for xh, xr in cache])) for b in grid]
        bias = float(grid[int(np.argmin(errs))])
        print(f"bias-only arm: fitted {1000*bias:+.2f} HU on {len(cache)} pair(s) "
              f"by minimising LAA-950 error")

    torch.manual_seed(0)        # the flow decoder SAMPLES: fix the draw so the
                                # certificate is reproducible across arms
    arb = Arbiter(args.tau_delta, args.tau_rho, args.tau_s)
    rows, agg = [], {k: [] for k in
                     ("laa_ref", "laa_rec", "laa_thick", "p15_ref", "p15_rec",
                      "delta", "rho", "rho_struct", "s", "psnr")}
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            y, x = item[0], item[1]
            y = y[None].to(args.device); x = x[None].to(args.device)
            base = item[3][None].to(args.device) if len(item) > 3 else None
            lung = lung_t = None
            if args.lung_mask:
                mp = ds.files[i].replace("_thin.npy", "_lung.npy")
                if os.path.exists(mp):
                    import numpy as _np
                    g = ds._crop_origin(_np.load(mp, mmap_mode="r").shape, i,
                                        _np.random.default_rng(i + 12345))
                    pd_, ph_, pw_ = ds.patch
                    lab = _np.load(mp, mmap_mode="r")[g[0]:g[0]+pd_, g[1]:g[1]+ph_,
                                                      g[2]:g[2]+pw_]
                    lung = torch.from_numpy(_np.ascontiguousarray(lab))[None, None].to(args.device)
                    lung_t = torch.nn.functional.max_pool3d(
                        (lung > 0).float(), kernel_size=(args.downsample, 1, 1),
                        stride=(args.downsample, 1, 1)) > 0.5
            out = model(y, base=base) if base is not None else model(y)
            if lung is not None:
                out = dict(out)
                out.update(model.densitometry(out["x_hat"], y, out["w_hat"], lung_t))
            if args.samples > 1 and args.flow and not args.oracle:
                acc = {k: [] for k in ("laa", "p15", "delta", "rho_struct", "s", "psnr")}
                for _ in range(args.samples):
                    o = model(y, base=base) if base is not None else model(y)
                    acc["laa"].append(metrics.laa950(o["x_hat"]).item())
                    acc["p15"].append(metrics.perc15(o["x_hat"]).item())
                    acc["delta"].append(float(o["delta_hat"][0]))
                    acc["rho_struct"].append(float(o["rho_struct"][0]))
                    acc["s"].append(float(o["s"][0]))
                    acc["psnr"].append(metrics.psnr(o["x_hat"], x).item())
                out = dict(out)
                out["_laa"] = float(np.mean(acc["laa"]))
                out["_p15"] = float(np.mean(acc["p15"]))
                out["_psnr"] = float(np.mean(acc["psnr"]))
                out["delta_hat"] = torch.tensor([np.mean(acc["delta"])], device=y.device)
                out["rho_struct"] = torch.tensor([np.mean(acc["rho_struct"])], device=y.device)
                out["s"] = [float(np.mean(acc["s"]))]
            if args.oracle:                       # perfect reconstruction
                out = dict(out)
                out["x_hat"] = x
                cert = model.densitometry(x, y, out["w_hat"])
                out.update(cert)
            elif bias:
                out = dict(out)
                out["x_hat"] = out["x_hat"] + bias
                out.update(model.densitometry(out["x_hat"], y, out["w_hat"]))
            xh = out["x_hat"]
            y_up = model.op.upsample_to_grid(y, xh.shape[2])
            v = arb({k: out[k] for k in ("delta_hat", "rho_struct", "s", "abstain")})[0]
            row = dict(case=os.path.basename(ds.files[i]),
                       w_hat=float(out["w_hat"][0]),
                       rho=float(out["rho"][0]),
                       delta_HU=1000 * float(out["delta_hat"][0]),
                       rho_struct=float(out["rho_struct"][0]),
                       s=("" if out["s"][0] != out["s"][0] else round(float(out["s"][0]), 3)),
                       verdict=v[0], violated=v[1] or "",
                       laa_ref=metrics.laa950(x, (lung > 0) if lung is not None else None).item(),
                       laa_rec=out.get("_laa", metrics.laa950(
                           xh, (lung > 0) if lung is not None else None).item()),
                       laa_thick=metrics.laa950(y_up, (lung > 0) if lung is not None else None).item(),
                       p15_ref_HU=1000 * metrics.perc15(x, (lung > 0) if lung is not None else None).item(),
                       p15_rec_HU=1000 * out.get("_p15", metrics.perc15(
                           xh, (lung > 0) if lung is not None else None).item()),
                       psnr=out.get("_psnr", metrics.psnr(xh, x).item()))
            lm = (lung > 0) if lung is not None else None
            row.update(
                laa910_ref=metrics.laa910(x, lm).item(),
                laa910_rec=metrics.laa910(xh, lm).item(),
                laa910_thick=metrics.laa910(y_up, lm).item(),
                p10_ref_HU=1000 * metrics.percN(x, 0.10, lm).item(),
                p10_rec_HU=1000 * metrics.percN(xh, 0.10, lm).item(),
                p05_ref_HU=1000 * metrics.percN(x, 0.05, lm).item(),
                p05_rec_HU=1000 * metrics.percN(xh, 0.05, lm).item())
            if args.per_lobe and lung is not None:
                for li, ln in enumerate(["UL_L", "LL_L", "UL_R", "ML_R", "LL_R"], 1):
                    sel = (lung == li)
                    if sel.sum() > 64:
                        row["laa950_ref_" + ln] = metrics.laa950(x, sel).item()
                        row["laa950_rec_" + ln] = metrics.laa950(xh, sel).item()
                    else:
                        row["laa950_ref_" + ln] = float("nan")
                        row["laa950_rec_" + ln] = float("nan")
            rows.append(row)
            for k, kk in (("laa_ref", "laa_ref"), ("laa_rec", "laa_rec"),
                          ("laa_thick", "laa_thick"), ("p15_ref", "p15_ref_HU"),
                          ("p15_rec", "p15_rec_HU"), ("delta", "delta_HU"),
                          ("rho", "rho"), ("rho_struct", "rho_struct"), ("psnr", "psnr")):
                agg[k].append(row[kk])
            if row["s"] != "":
                agg["s"].append(row["s"])

    A = {k: np.array(v, dtype=float) for k, v in agg.items() if len(v)}
    ref, rec, thick = A["laa_ref"], A["laa_rec"], A["laa_thick"]
    cert_mask = np.array([r["verdict"] == "certified" for r in rows])
    print(f"\nn={len(rows)}   {'ORACLE (x_hat := true thin)' if args.oracle else args.ckpt}")
    print(f"LAA-950 MAE  thick   : {np.mean(np.abs(thick-ref)):.2f} pp")
    print(f"LAA-950 MAE  model   : {np.mean(np.abs(rec-ref)):.2f} pp")
    print(f"LAA-950 CCC  model   : "
          f"{metrics.ccc(torch.tensor(rec), torch.tensor(ref)).item():.3f}")
    print(f"Perc15  MAE  model   : {np.mean(np.abs(A['p15_rec']-A['p15_ref'])):.2f} HU")
    print(f"PSNR                 : {np.mean(A['psnr']):.2f} dB")
    print(f"certificate  delta   : {np.mean(A['delta']):+.2f} +- {np.std(A['delta']):.2f} HU"
          f"   (|delta| mean {np.mean(np.abs(A['delta'])):.2f})")
    print(f"certificate  rho     : {np.mean(A['rho']):.3f}   rho_struct "
          f"{np.mean(A['rho_struct']):.3f}")
    if "s" in A:
        print(f"certificate  s       : {np.mean(A['s']):+.2f} +- {np.std(A['s']):.2f} "
              f"(units of sigma_v)")
    print(f"verdicts             : certified {int(cert_mask.sum())}/{len(rows)}"
          + (f"   LAA MAE on certified {np.mean(np.abs(rec-ref)[cert_mask]):.2f} pp"
             if cert_mask.any() else ""))

    for k, lab in (("laa910", "LAA-910 MAE  model  "), ("p10", "Perc10  MAE  model  "),
                   ("p05", "Perc5   MAE  model  ")):
        r_ = np.array([x_[k + "_ref" + ("_HU" if k.startswith("p") else "")] for x_ in rows])
        m_ = np.array([x_[k + "_rec" + ("_HU" if k.startswith("p") else "")] for x_ in rows])
        unit = "HU" if k.startswith("p") else "pp"
        print(f"{lab} : {np.mean(np.abs(m_ - r_)):.2f} {unit}")
    if args.per_lobe and any("laa950_ref_UL_L" in r_ for r_ in rows):
        print("per-lobe LAA-950 MAE (pp):", end=" ")
        for ln in ["UL_L", "LL_L", "UL_R", "ML_R", "LL_R"]:
            a_ = np.array([r_["laa950_ref_" + ln] for r_ in rows], dtype=float)
            b_ = np.array([r_["laa950_rec_" + ln] for r_ in rows], dtype=float)
            ok = ~(np.isnan(a_) | np.isnan(b_))
            print(f"{ln} {np.mean(np.abs(b_-a_)[ok]):.2f}", end="  ")
        print()
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
