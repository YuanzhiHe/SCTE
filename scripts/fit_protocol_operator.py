"""Fit the PROTOCOL-level acquisition constants ARC-CT needs, from real pairs.

Two quantities, both protocol-scoped, both fitted offline on paired data and
shipped together (scte_r/agentic.TailCalibration):

  protocol_w  the effective through-plane width of this protocol's thick series.
              Fitted with the TRUE thin volume as reference, which is legitimate
              here (offline, on calibration pairs) and NOT available at inference
              — the per-scan fit is degenerate (see agentic.OperatorSolver).

  tail relation  ridge regression predicting the thin-slice through-plane lung
              variance from quantities observable at inference
              (Var_z(y), w, sigma_hat, Var_z(y)*w), with sigma_v the HELD-OUT
              residual spread under 5-fold grouping BY CASE. The term is marked
              `informative` only when sigma_v is smaller than the between-case
              spread of the target itself; otherwise ARC-CT reports the
              tail-recovery term as UNAVAILABLE rather than shipping a
              calibration that carries no information.

Usage
-----
  python scripts/fit_protocol_operator.py --root DATA/public_pairs \
         --protocol "RPLHR-CT/unknown-kernel/5mm/r5" --out runs/protocol_rplhr_5mm.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.agentic import lung_set, z_variance
from scte_r.datasets import PreparedPairDataset
from scte_r.forward_operator import SSPForwardOperator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="prepared real pairs (prep_pairs.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default="unspecified",
                    help="(manufacturer/model, kernel, nominal thickness, r) tag this "
                         "calibration is valid for")
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--patch", type=int, nargs=3, default=[60, 160, 160])
    ap.add_argument("--n_pairs", type=int, default=0, help="0 = all cases in --root")
    ap.add_argument("--grid", type=float, nargs=3, default=[2.0, 12.0, 0.25])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ds = PreparedPairDataset(args.root, patch=tuple(args.patch),
                             downsample=args.downsample)
    n = len(ds) if not args.n_pairs else min(args.n_pairs, len(ds))
    grid = np.arange(args.grid[0], args.grid[1] + 1e-9, args.grid[2])
    r = args.downsample

    ws, feats, targets, names = [], [], [], []
    for i in range(n):
        y, x, _ = ds[i]
        y = y[None].to(args.device); x = x[None].to(args.device)
        errs = [float(((SSPForwardOperator(slice_fwhm_mm=float(w), downsample=r)(x) - y) ** 2).mean())
                for w in grid]
        w_i = float(grid[int(np.argmin(errs))])
        vzy = float(z_variance(y, lung_set(y)))
        vzx = float(z_variance(x, lung_set(x)))
        dz = y[..., 1:, :, :] - y[..., :-1, :, :]
        sig = float(dz.flatten(1).std(dim=1)[0]) / 2 ** 0.5
        if np.isnan(vzy) or np.isnan(vzx):
            print(f"[skip] {os.path.basename(ds.files[i])}: empty lung set"); continue
        ws.append(w_i); targets.append(vzx); names.append(os.path.basename(ds.files[i]))
        feats.append([vzy, w_i, sig, vzy * w_i])
        print(f"[{i + 1:3d}/{n}] {names[-1]:<24} w={w_i:5.2f} mm  "
              f"var_z(y)={1e6*vzy:8.1f}  var_z(x)={1e6*vzx:8.1f} HU^2")

    if len(ws) < 3:
        # Two very different causes, and telling them apart on site saves an hour:
        # nothing usable => the data is probably not in HU; some usable but < 3 =>
        # the cohort (or this half of the split) is simply too small to fit on.
        if len(ws) == 0:
            sys.exit(f"\nERROR: 0 usable case(s) out of {n} — every case had an empty lung\n"
                     "set, which almost always means the pairs are not in Hounsfield units\n"
                     "(check the HU range printed by prep_pairs.py: air ~ -1000, soft tissue\n"
                     "~ 0). Re-run prep_pairs.py with --rescale_hu SLOPE INTERCEPT (12-bit CT\n"
                     "packing is usually 3072 -1024), then re-run this step. Nothing written.")
        sys.exit(f"\nERROR: only {len(ws)} usable case(s) out of {n} — need at least 3.\n"
                 f"The {len(ws)} that worked look fine (widths {', '.join(f'{w:.2f}' for w in ws)} mm),\n"
                 "so this is a cohort-size problem, NOT a units problem: --root points at a\n"
                 "directory with too few pairs (in run_private.sh this is the TRAIN half, so\n"
                 "the cohort needs >= 6 cases). Nothing has been written.")
    ws = np.array(ws); X = np.array(feats); t = np.array(targets)
    print(f"\nprotocol width: median {np.median(ws):.2f} mm, mean {ws.mean():.2f} "
          f"+- {ws.std():.2f} (n={len(ws)}), range [{ws.min():.2f}, {ws.max():.2f}]")

    # grouped-by-case CV: every case is its own group, so no case is in both folds
    idx = np.arange(len(t)); rng = np.random.default_rng(0); rng.shuffle(idx)
    folds = np.array_split(idx, args.folds)
    resid = []
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xt = np.c_[X[tr], np.ones(len(tr))]
        A = Xt.T @ Xt + args.ridge * np.eye(Xt.shape[1])
        beta = np.linalg.solve(A, Xt.T @ t[tr])
        pred = np.c_[X[f], np.ones(len(f))] @ beta
        resid.extend(pred - t[f])
    sigma_v = float(np.std(resid))
    between = float(np.std(t))
    informative = sigma_v < between
    Xa = np.c_[X, np.ones(len(t))]
    beta = np.linalg.solve(Xa.T @ Xa + args.ridge * np.eye(Xa.shape[1]), Xa.T @ t)
    print(f"tail relation: held-out residual sd {1e6*sigma_v:.1f} HU^2 vs between-case "
          f"sd {1e6*between:.1f} HU^2 -> informative={informative}")
    if not informative:
        print("  -> the tail-recovery term will be reported UNAVAILABLE (honest refusal)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump({"protocol": args.protocol,
               "protocol_w": float(np.median(ws)),
               "protocol_w_sd": float(ws.std()),
               "n_pairs": int(len(ws)),
               "coef": beta[:-1].tolist(), "intercept": float(beta[-1]),
               "sigma_v": sigma_v, "between_case_sd": between,
               "informative": bool(informative),
               "features": ["var_z_thick", "w", "sigma_hat", "var_z_thick_x_w"]},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
