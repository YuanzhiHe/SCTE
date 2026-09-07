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
import argparse, csv, gc, os, sys
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
ap.add_argument('--save_std', action='store_true',
                help='with --samples N, also write <case>_std.npy (per-voxel spread '
                     'across samples) and keep sample 1 as the reconstruction. A real '
                     'structure is pinned down by the measurement and should repeat; a '
                     'fabricated one is a draw from the sampler and should not. Unlike '
                     'the data-consistency residual, this does not require the error to '
                     'violate the observation - which is exactly why that one failed.')
ap.add_argument('--save_recon', default=None, metavar='DIR',
                help='write <case>_rec.npy so the reconstruction can be audited for '
                     'fabricated / erased lesions (scripts/hallucination.py)')
ap.add_argument('--paren_gate', type=float, default=None, metavar='HU',
                help='inject the generative residual only in lung PARENCHYMA: inside the '
                     'lung and below this HU on the backbone reconstruction, so vessels '
                     'and airway walls are left to the backbone. Gating by DIFFICULTY '
                     '(--grad_gate) does the opposite and was measured to halve the '
                     'LAA concordance: the hardest voxels (vessels, gradient ~125 HU) '
                     'carry no LAA signal, and the voxels that do (parenchyma, gradient '
                     '~43 HU) are the ones difficulty-gating discards.')
ap.add_argument('--grad_gate', type=float, default=0.0, metavar='Q',
                help='inject the generative residual only where the OBSERVABLE '
                     'through-plane gradient (from the thick series, so available at '
                     'inference) is above its Q-th percentile inside the lung. '
                     'Measured: in the flattest 20%% of voxels the sampler is 2%% WORSE '
                     'than the backbone, while the top 20%% carry 56%% of the benefit.')
ap.add_argument('--grad_steps', default=None, metavar='LO,HI',
                help='per-slab adaptive Euler steps: LO steps for slabs whose '
                     'high-gradient fraction is small, HI for the rest')
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
ap.add_argument('--start', type=int, default=0,
                help='skip the first N cases; with --csv, rows already on disk are kept')
ap.add_argument('--log', default=None, metavar='FILE',
                help='tee all stdout/stderr to this file (append) so output survives '
                     'even when launched from a process whose own redirection is blocked')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
if a.log:
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, data):
            for s in self.streams:
                try: s.write(data)
                except Exception: pass
        def flush(self):
            for s in self.streams:
                try: s.flush()
                except Exception: pass
    _lf = open(a.log, 'a', buffering=1)
    sys.stdout = _Tee(_lf, sys.__stdout__)   # file FIRST: a dead console must not
    sys.stderr = _Tee(_lf, sys.__stderr__)   # block the log that diagnosing depends on
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

def z_gradient(thick_hu, out_depth, op):
    """|d/dz| of the THICK series on the thin grid - an inference-time difficulty map.

    It explains the per-structure error almost exactly (pulmonary veins MAE 128 HU at
    gradient 129; aorta 18 HU at gradient 22), which is why this is a gradient gate and
    not an anatomy gate: 'hard' means 'changes fast through-plane', and vessels are hard
    because they run obliquely, not because they are vessels.
    """
    g = np.abs(np.diff(thick_hu, axis=0, prepend=thick_hu[:1]))
    return op.upsample_to_grid(torch.from_numpy(g[None, None] / 1000.),
                               out_depth)[0, 0].numpy() * 1000.


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
done = set()
if a.csv and os.path.exists(a.csv):
    # identity-based resume: whatever case ids are already in the CSV are skipped,
    # regardless of --start. Position-based resume silently dropped cases when the
    # skip count and the CSV row count ever disagreed (observed: --start 10 with an
    # 8-row CSV -> cases 9-10 would have been lost from the final table).
    with open(a.csv) as fh:
        for row in csv.DictReader(fh):
            rows.append(row); done.add(row['case'])
    print('resuming: %d cases already in %s' % (len(done), a.csv), flush=True)
cases = [f for f in cases if f[:-9] not in done]
print('%d cases to run (%d done)' % (len(cases), len(done)), flush=True)

def flush_csv():
    if a.csv and rows:
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        # heartbeat file: python's own writes are the one I/O path proven reliable
        # here, so progress is reported through it even if stdout logging fails
        with open(os.path.splitext(a.csv)[0] + '.progress', 'w') as fh:
            fh.write('%d\n' % len(rows))
for ci, f in enumerate(cases, 1):
    if a.device == 'cuda':        # cached blocks from earlier cases accumulate until the
        torch.cuda.empty_cache()  # 6 GB WDDM driver starts paging and every slab crawls;
        gc.collect()              # releasing between cases keeps each one at full speed
    thin = np.load(os.path.join(a.root, f))
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')))
    mp = os.path.join(a.root, f.replace('_thin', '_lung'))
    has_mask = os.path.exists(mp)
    if not has_mask and not a.save_recon:
        print('[skip] %s: no lung mask' % f); continue
    # Reconstructing needs no lung mask - it only enters the metrics. Cohorts scored by
    # lesion survival rather than densitometry (LUNA16) have no reason to carry one.
    lung = np.load(mp) if has_mask else np.zeros(0, np.uint8)
    base_vol = None
    if a.base_recon:
        bp = os.path.join(a.root, f.replace('_thin', '_base'))
        if not os.path.exists(bp):
            print('[skip] %s: no cached base reconstruction' % f); continue
        base_vol = np.load(bp).astype(np.float32)
    D = thin.shape[0]
    # Per-case slab size: the per-slab GPU footprint scales with in-plane area
    # (activations are channels x slab x H x W), so a 512x512 series needs a
    # shallower slab than a 280x300 one or the 6 GB WDDM driver starts paging
    # (observed on a 512x512 case: fb 5784/6113 MiB, sm 100%, memory controller
    # 0% - compute halted). 2.6M voxels/slab is the largest configuration that
    # ran with headroom; smaller in-plane keeps the full requested slab.
    slab, ov = a.slab, a.overlap
    if thin.shape[1] * thin.shape[2] * slab > 2_600_000:
        slab = max(2 * r, int(2_600_000 / (thin.shape[1] * thin.shape[2])) // r * r)
        ov = max(r, slab // 2 // r * r)
    grad = None
    if a.grad_gate or a.grad_steps:
        from scte_r.forward_operator import SSPForwardOperator
        grad = z_gradient(thick.astype(np.float32), thin.shape[0],
                          SSPForwardOperator(slice_fwhm_mm=5.0, downsample=r))
    if a.interp:                                  # classical baseline: one matmul, no slabs
        W = interp_matrix(D, thick.shape[0], r, a.interp)
        rec = np.tensordot(W, thick.astype(np.float32), axes=(1, 0))
        starts = []
    else:
        rec = None
    gthr = None; slab_steps = []; sacc = None
    if grad is not None:
        # threshold inside a REFERENCE-FREE lung region: the stored _lung.npy comes
        # from TotalSegmentator run on the 1 mm reference, which is not available when
        # this actually runs on a 5 mm-only archive.
        lm = (lung_from(base_vol, 2) if base_vol is not None
              else ((lung > 0) if has_mask else np.ones(thin.shape, bool)))
        gthr = float(np.percentile(grad[lm], a.grad_gate if a.grad_gate else 80.0))
    acc = np.zeros(thin.shape, np.float32); wacc = np.zeros(D, np.float32) + 1e-8
    step = slab - ov
    starts = list(range(0, max(D - slab, 0) + 1, step))
    if starts[-1] + slab < D: starts.append(D - slab)
    with torch.no_grad():
        for z0 in (starts if rec is None else []):
            z0 -= z0 % r
            yt = torch.from_numpy(thick[z0 // r: z0 // r + slab // r].astype(np.float32)
                                  )[None, None].to(a.device) / 1000.
            if yt.shape[2] * r != slab: continue
            bs = None if base_vol is None else torch.from_numpy(
                base_vol[z0:z0 + slab])[None, None].to(a.device) / 1000.
            w_hann = hann_z(slab, ov)
            steps_here = None
            if a.grad_steps and grad is not None:
                lo, hi = (int(v) for v in a.grad_steps.split(','))
                gs = grad[z0:z0 + slab]
                frac = float((gs > gthr).mean()) if gthr is not None else 1.0
                steps_here = hi if frac > 0.05 else lo
                slab_steps.append(steps_here)
                model.steps = steps_here
            if a.samples <= 1:
                xs = model(yt)['x_hat'] if bs is None else model(yt, base=bs)['x_hat']
            else:
                s1 = ssum = ssq = None
                for si in range(a.samples):
                    o = model(yt)['x_hat'] if bs is None else model(yt, base=bs)['x_hat']
                    if si == 0: s1 = o
                    ssum = o if ssum is None else ssum + o
                    ssq = o * o if ssq is None else ssq + o * o
                mean = ssum / a.samples
                if a.save_std:
                    var = (ssq / a.samples - mean * mean).clamp_min(0)
                    sd = var.sqrt()
                    if sacc is None:
                        sacc = np.zeros(thin.shape, np.float32)
                    sacc[z0:z0 + slab] += sd[0, 0].cpu().numpy() * 1000. * w_hann[:, None, None]
                    xs = s1                       # audit sample 1, not the mean
                else:
                    xs = mean
            w_hann = hann_z(slab, ov)
            acc[z0:z0 + slab] += xs[0, 0].cpu().numpy() * 1000. * w_hann[:, None, None]
            wacc[z0:z0 + slab] += w_hann
    if rec is None: rec = acc / wacc[:, None, None]
    if a.lung_blend and base_vol is not None:
        bm = lung_from(base_vol, a.lung_blend)
        if a.grad_gate and grad is not None:
            bm &= grad > gthr        # and only where it is actually hard
        if a.paren_gate is not None:
            bm &= base_vol < a.paren_gate     # gate by CONSEQUENCE, not difficulty
        rec = np.where(bm, rec, base_vol)                 # backbone keeps the rest
    if a.save_recon:
        os.makedirs(a.save_recon, exist_ok=True)
        np.save(os.path.join(a.save_recon, f.replace('_thin', '_rec')),
                rec.astype(np.float16))
        if sacc is not None:
            np.save(os.path.join(a.save_recon, f.replace('_thin', '_std')),
                    (sacc / wacc[:, None, None]).astype(np.float16))
    if not has_mask:
        print('[%2d/%d] %-16s cached (no mask -> no metrics)'
              % (ci, len(cases), row_case if False else f[:-9]), flush=True)
        continue
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
    ss = ('  steps %d-%d avg %.0f' % (min(slab_steps), max(slab_steps),
          np.mean(slab_steps))) if slab_steps else ''
    sm = ('  slab %d/%d' % (slab, ov)) if slab != a.slab else ''
    gm = ('  GPU %.1f/%.1f GB' % (torch.cuda.memory_allocated() / 2**30,
          torch.cuda.memory_reserved() / 2**30)) if a.device == 'cuda' else ''
    print('[%2d/%d] %-14s LAA950 ref %6.2f%% rec %6.2f%%  PSNR %5.2f  lungPSNR %5.2f  SSIM %.4f%s%s%s'
          % (ci, len(cases), row['case'], row['laa950_ref'], row['laa950_rec'],
             pw, pl, sw, ss, sm, gm), flush=True)
    flush_csv()

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
