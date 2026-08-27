"""Whole-volume sliding-window inference and WHOLE-LUNG densitometry.

Everything up to here was patch-level: one (40,64,64) box per case, chosen to be
lung-dense. That is the right unit for studying the MECHANISM but it is not the
clinical quantity — LAA-950 is read over the whole lung, and a patch cannot show
whether errors cancel or accumulate across lobes. This script runs the decoder in
z-slabs across the entire volume, blends them with a Hann ramp, and reports
whole-lung and per-lobe densitometry against the 1 mm reference.

  python scripts/infer_volume.py --root DATA/public_pairs_test --ckpt runs/FL_sr0.028.pt \
      --flow --residual_scale 0.028 --flow_steps 32 --learned_op runs/forward_op_rplhr.pt \
      --calibration runs/protocol_rplhr_5mm.json --n 15 --csv results/volume_flow.csv
"""
import argparse, csv, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r import metrics
from scte_r.agentic import TailCalibration
from scte_r.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--ckpt', default=None)
ap.add_argument('--flow', action='store_true'); ap.add_argument('--residual_scale', type=float, default=0.028)
ap.add_argument('--flow_steps', type=int, default=32)
ap.add_argument('--base_ch', type=int, default=None); ap.add_argument('--n_blocks', type=int, default=None)
ap.add_argument('--learned_op', default=None); ap.add_argument('--calibration', default=None)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--slab', type=int, default=40, help='thin slices per slab (multiple of r)')
ap.add_argument('--overlap', type=int, default=20, help='thin-slice overlap between slabs')
ap.add_argument('--trilinear', action='store_true', help='alias for --interp linear')
ap.add_argument('--interp', default=None,
                choices=['nearest', 'linear', 'cubic', 'lanczos'],
                help='classical z-interpolation baseline instead of a model')
ap.add_argument('--data_range', type=float, default=2000.0,
                help='HU range for PSNR/SSIM (2000 = the [-1000,1000] convention used '
                     'throughout this repo; pass 4095 for the 12-bit convention)')
ap.add_argument('--lung_blend', type=int, default=0, metavar='DILATE',
                help='inject the generative residual only inside the lung, dilated by '
                     'DILATE voxels. The mask is derived from the BASE RECONSTRUCTION '
                     '(available at inference), never from the reference - outside the '
                     'lung the backbone is already excellent and the sampler only adds '
                     'noise there, which is where most of the whole-image PSNR is lost.')
ap.add_argument('--samples', type=int, default=1,
                help='average N stochastic samples. Adding tail texture and maximising '
                     'PSNR are mathematically opposed, so averaging trades densitometry '
                     'back for image fidelity: use >1 when the output is for viewing.')
ap.add_argument('--base_recon', action='store_true',
                help='feed the cached <case>_base.npy strong reconstruction to the '
                     'flow decoder as its base predictor')
ap.add_argument('--n', type=int, default=0); ap.add_argument('--csv', default=None)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
if a.trilinear and not a.interp: a.interp = 'linear'
r = a.downsample
assert a.slab % r == 0 and a.overlap % r == 0, 'slab and overlap must be multiples of r'

lop = None
if a.learned_op:
    from scte_r.forward_learned import LearnedOperator
    d = torch.load(a.learned_op, map_location=a.device, weights_only=False)
    lop = LearnedOperator(d['downsample'], d['widths'], correction=d['correction'])
    lop.load_state_dict(d['state_dict']); lop = lop.to(a.device).eval()
    for q in lop.parameters(): q.requires_grad_(False)

def interp_matrix(D, Dt, r, kind):
    """z-only interpolation, with the CORRECT node positions.

    Thick slice k is the average of thin slices [rk, rk+r-1], so it samples the
    thin grid at rk+(r-1)/2 - NOT at rk. Getting this wrong costs half a slice of
    systematic z-shift and would quietly handicap every classical baseline.
    """
    pz = r * np.arange(Dt) + (r - 1) / 2.0
    j = np.arange(D, dtype=float)
    if kind == 'lanczos':                       # a=3 windowed sinc
        x = (j[:, None] - pz[None, :]) / r
        L = np.sinc(x) * np.sinc(x / 3.0); L[np.abs(x) >= 3.0] = 0.0
        ssum = L.sum(1, keepdims=True); ssum[ssum == 0] = 1.0
        return (L / ssum).astype(np.float32)
    from scipy.interpolate import interp1d, CubicSpline
    I = np.eye(Dt)
    if kind == 'cubic':                          # natural cubic spline == B-spline interp
        W = CubicSpline(pz, I, axis=0, bc_type='natural')(j)
    else:
        W = interp1d(pz, I, axis=0, kind=kind, bounds_error=False,
                     fill_value=(I[0], I[-1]))(j)
    return W.astype(np.float32)


def image_quality(rec, ref, mask, data_range):
    """PSNR/SSIM over the whole prepared volume and over the lung only.

    NOTE: the prepared volumes are already cropped to the lung bounding box, so
    'whole' here excludes the air background that inflates full-image PSNR in the
    super-resolution literature. These numbers are therefore LOWER than, and not
    comparable to, whole-CT PSNR reported elsewhere - but they are comparable
    between the arms below, which is what they are for.
    """
    from skimage.metrics import structural_similarity
    mse = float(np.mean((rec - ref) ** 2))
    psnr_w = 10 * np.log10(data_range ** 2 / max(mse, 1e-8))
    mse_l = float(np.mean((rec[mask] - ref[mask]) ** 2))
    psnr_l = 10 * np.log10(data_range ** 2 / max(mse_l, 1e-8))
    _, smap = structural_similarity(ref, rec, data_range=data_range, full=True)
    return psnr_w, float(smap.mean()), psnr_l, float(smap[mask].mean())


model = None
if not a.interp:
    mkw = dict(iters=1, downsample=r, slice_fwhm_mm=5.0, learned_op=lop,
               calibration=TailCalibration.load(a.calibration))
    if a.flow: mkw.update(flow=True, residual_scale=a.residual_scale, steps=a.flow_steps)
    else:      mkw.update(arc=True)
    if a.base_ch:  mkw['base_ch'] = a.base_ch
    if a.n_blocks: mkw['n_blocks'] = a.n_blocks
    model = build_model(**mkw).to(a.device).eval()
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location=a.device)
        if not any(k.startswith('net.') for k in sd): sd = {f'net.{k}': v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

def lung_from(vol_hu, dilate):
    """Reference-free lung mask: air-window threshold on the reconstruction itself,
    largest connected component, hole-filled, then dilated so no seam lands on a
    lung boundary. Coarser than TotalSegmentator, which is the point - it must be
    computable from what the scanner gives us at inference time."""
    from scipy import ndimage
    m = (vol_hu > -1024) & (vol_hu < -400)
    m = ndimage.binary_closing(m, np.ones((3, 3, 3)))
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.argsort(sizes)[::-1][:2] + 1            # two lungs
        m = np.isin(lab, keep[sizes[keep - 1] > 0.05 * sizes.max()])
    for z in range(m.shape[0]):
        m[z] = ndimage.binary_fill_holes(m[z])
    if dilate > 0:
        m = ndimage.binary_dilation(m, ndimage.generate_binary_structure(3, 1),
                                    iterations=dilate)
    return m


def hann_z(n, ov):
    w = np.ones(n, np.float32)
    if ov > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(ov) / max(ov - 1, 1))
        w[:ov] = ramp; w[-ov:] = ramp[::-1]
    return w

cases = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.n: cases = cases[:a.n]
torch.manual_seed(0)
rows = []
for ci, f in enumerate(cases, 1):
    thin = np.load(os.path.join(a.root, f))
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')))
    mp = os.path.join(a.root, f.replace('_thin', '_lung'))
    if not os.path.exists(mp):
        print('[skip] %s: no lung mask' % f); continue
    lung = np.load(mp)
    base_vol = None
    if a.base_recon:
        bp = os.path.join(a.root, f.replace('_thin', '_base'))
        if not os.path.exists(bp):
            print('[skip] %s: no cached base reconstruction' % f); continue
        base_vol = np.load(bp).astype(np.float32)
    D = thin.shape[0]
    if a.interp:                                  # classical baseline: one matmul, no slabs
        W = interp_matrix(D, thick.shape[0], r, a.interp)
        rec = np.tensordot(W, thick.astype(np.float32), axes=(1, 0))
        starts = []
    else:
        rec = None
    acc = np.zeros(thin.shape, np.float32); wacc = np.zeros(D, np.float32) + 1e-8
    step = a.slab - a.overlap
    starts = list(range(0, max(D - a.slab, 0) + 1, step))
    if starts[-1] + a.slab < D: starts.append(D - a.slab)
    with torch.no_grad():
        for z0 in (starts if rec is None else []):
            z0 -= z0 % r
            yt = torch.from_numpy(thick[z0 // r: z0 // r + a.slab // r].astype(np.float32)
                                  )[None, None].to(a.device) / 1000.
            if yt.shape[2] * r != a.slab: continue
            bs = None if base_vol is None else torch.from_numpy(
                base_vol[z0:z0 + a.slab])[None, None].to(a.device) / 1000.
            if a.samples <= 1:
                xs = model(yt)['x_hat'] if bs is None else model(yt, base=bs)['x_hat']
            else:
                acc_s = None
                for _ in range(a.samples):
                    o = model(yt)['x_hat'] if bs is None else model(yt, base=bs)['x_hat']
                    acc_s = o if acc_s is None else acc_s + o
                xs = acc_s / a.samples
            w = hann_z(a.slab, a.overlap)
            acc[z0:z0 + a.slab] += xs[0, 0].cpu().numpy() * 1000. * w[:, None, None]
            wacc[z0:z0 + a.slab] += w
    if rec is None: rec = acc / wacc[:, None, None]
    if a.lung_blend and base_vol is not None:
        bm = lung_from(base_vol, a.lung_blend)
        rec = np.where(bm, rec, base_vol)                 # backbone keeps the rest
    m = lung > 0
    if m.sum() < 1000: print('[skip] %s: empty lung' % f); continue
    t = lambda v: torch.from_numpy(v.astype(np.float32) / 1000.)
    mt = torch.from_numpy(m)
    pw, sw, pl, sl = image_quality(rec, thin.astype(np.float32), m, a.data_range)
    row = dict(case=f.replace('_thin.npy', ''), lung_voxels=int(m.sum()),
               psnr=pw, ssim=sw, lung_psnr=pl, lung_ssim=sl,
               laa950_ref=metrics.laa950(t(thin), mt).item(),
               laa950_rec=metrics.laa950(t(rec), mt).item(),
               laa910_ref=metrics.laa910(t(thin), mt).item(),
               laa910_rec=metrics.laa910(t(rec), mt).item(),
               p15_ref=1000 * metrics.percN(t(thin), .15, mt).item(),
               p15_rec=1000 * metrics.percN(t(rec), .15, mt).item(),
               p10_ref=1000 * metrics.percN(t(thin), .10, mt).item(),
               p10_rec=1000 * metrics.percN(t(rec), .10, mt).item())
    for li, ln in enumerate(['UL_L', 'LL_L', 'UL_R', 'ML_R', 'LL_R'], 1):
        sel = torch.from_numpy(lung == li)
        row['laa950_ref_' + ln] = metrics.laa950(t(thin), sel).item() if sel.sum() > 1000 else float('nan')
        row['laa950_rec_' + ln] = metrics.laa950(t(rec), sel).item() if sel.sum() > 1000 else float('nan')
    rows.append(row)
    print('[%2d/%d] %-14s LAA950 ref %6.2f%% rec %6.2f%%  PSNR %5.2f  lungPSNR %5.2f  SSIM %.4f'
          % (ci, len(cases), row['case'], row['laa950_ref'], row['laa950_rec'],
             pw, pl, sw), flush=True)

A = lambda k: np.array([x[k] for x in rows], dtype=float)
print('\nn=%d  WHOLE-LUNG (TotalSegmentator mask)' % len(rows))
print('  image quality (data_range=%.0f HU; add %+.2f dB for the 4095 convention)'
      % (a.data_range, 20 * np.log10(4095.0 / a.data_range)))
for k in ('psnr', 'lung_psnr', 'ssim', 'lung_ssim'):
    v = A(k); print('  %-10s %8.4f ± %.4f' % (k, v.mean(), v.std()))
for k, u in (('laa950', 'pp'), ('laa910', 'pp'), ('p15', 'HU'), ('p10', 'HU')):
    ref, rec = A(k + '_ref'), A(k + '_rec')
    ccc = metrics.ccc(torch.tensor(rec), torch.tensor(ref)).item()
    print('  %-7s ref %7.2f±%5.2f | rec %7.2f±%5.2f | bias %+6.2f | MAE %5.2f %s | CCC %.3f'
          % (k, ref.mean(), ref.std(), rec.mean(), rec.std(), (rec - ref).mean(),
             np.abs(rec - ref).mean(), u, ccc))
print('  per-lobe LAA950 MAE:', ' '.join(
    '%s %.2f' % (ln, np.nanmean(np.abs(A('laa950_rec_' + ln) - A('laa950_ref_' + ln))))
    for ln in ['UL_L', 'LL_L', 'UL_R', 'ML_R', 'LL_R']))
if a.csv:
    with open(a.csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', a.csv)
