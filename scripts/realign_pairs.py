"""Re-align already-prepared .npy pairs at THIN-slice resolution.

`prep_pairs.py` used to search the thin/thick offset in whole 5 mm slices only, so
a sub-slab shift could never be corrected. On the public cohort that left a
systematic 2-thin-slice offset between the forward operator's slabs and the thick
slice they are supposed to explain. Consequences, all measured:

  * forward-operator residual  0.1364 -> 0.0974   (-29%)
  * cubic interpolation PSNR   27.15  -> 30.22 dB (+3.07 dB)

so every classical baseline was handicapped and every data-consistency projection
was enforcing a shifted constraint. This script fixes existing prepared data
exactly - by re-slicing, never by interpolating - so nothing is invented.

For each case it searches the thin start index t0 and thick start index k0 that
minimise ||A_w(thin[t0:]) - thick[k0:]||, then writes the trimmed, aligned pair
(and the lung mask, trimmed identically).

  python scripts/realign_pairs.py --src DATA/public_pairs_test --dst DATA/aligned_test
"""
import argparse, os, shutil
import numpy as np, torch, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', default=None,
                help='omit together with --check to measure without writing anything')
ap.add_argument('--check', action='store_true',
                help='report the alignment each case WOULD need and exit non-zero if any '
                     'case needs a non-zero shift; writes nothing')
ap.add_argument('--limit', type=int, default=0, help='check only the first N cases')
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--slice_fwhm', type=float, default=6.25,
                help='protocol width used only for the alignment search')
ap.add_argument('--max_thick_shift', type=int, default=1)
a = ap.parse_args()
r = a.downsample
op = SSPForwardOperator(slice_fwhm_mm=a.slice_fwhm, downsample=r)
if not a.check:
    if not a.dst: ap.error('--dst is required unless --check is given')
    os.makedirs(a.dst, exist_ok=True)


def residual(thin, thick):
    n = thin.shape[0] // r
    n = min(n, thick.shape[0])
    if n < 4: return float('inf')
    sim = op(torch.from_numpy(thin[:n * r][None, None]) / 1000.)[0, 0].numpy()
    y = thick[:n] / 1000.
    return float(np.linalg.norm(sim - y) / max(np.linalg.norm(y - y.mean()), 1e-8))


files = sorted(f for f in os.listdir(a.src) if f.endswith('_thin.npy'))
if a.limit: files = files[:a.limit]
shifts, before, after = [], [], []
for f in files:
    thin = np.load(os.path.join(a.src, f))
    thick = np.load(os.path.join(a.src, f.replace('_thin', '_thick')))
    mp = os.path.join(a.src, f.replace('_thin', '_lung'))
    lung = np.load(mp) if os.path.exists(mp) else None
    # centre crop in-plane for the search: alignment is a z question, and the lung
    # interior carries the through-plane structure the operator is fitted on
    H, W = thin.shape[1:]
    ys = slice(H // 4, 3 * H // 4); xs = slice(W // 4, 3 * W // 4)
    base = residual(thin[:, ys, xs], thick[:, ys, xs])
    best, best_e = (0, 0), float('inf')
    for k0 in range(a.max_thick_shift + 1):
        for t0 in range(0, 2 * r):
            e = residual(thin[t0:, ys, xs], thick[k0:, ys, xs])
            if e < best_e: best, best_e = (k0, t0), e
    k0, t0 = best
    n = min((thin.shape[0] - t0) // r, thick.shape[0] - k0)
    if n < 4:
        print('[skip] %s: too short after alignment' % f); continue
    tn = thin[t0:t0 + n * r]; tk = thick[k0:k0 + n]
    if not a.check:
        np.save(os.path.join(a.dst, f), tn)
        np.save(os.path.join(a.dst, f.replace('_thin', '_thick')), tk)
        if lung is not None:
            np.save(os.path.join(a.dst, f.replace('_thin', '_lung')), lung[t0:t0 + n * r])
    shifts.append((k0, t0)); before.append(base); after.append(best_e)
    print('[ok] %-14s thick+%d thin+%d  residual %.4f -> %.4f  thin %s'
          % (f.replace('_thin.npy', ''), k0, t0, base, best_e, tn.shape), flush=True)

print('\nn=%d  residual %.4f -> %.4f  (%.0f%% lower)'
      % (len(after), np.mean(before), np.mean(after),
         100 * (1 - np.mean(after) / max(np.mean(before), 1e-9))))
u, c = np.unique([s[1] for s in shifts], return_counts=True)
print('thin-slice shifts chosen:', dict(zip(u.tolist(), c.tolist())))
# Only the SUB-SLAB part matters. (k0=1, t0=5) is the same alignment as (0, 0) with
# one whole slab dropped, and the search will pick it whenever trimming a boundary
# slab happens to score a hair lower. Testing the raw t0 would fail perfectly aligned
# data - the quantity that actually says "the operator grid is wrong" is t0 - r*k0.
sub = [(t0 - r * k0) % r for k0, t0 in shifts]
us, cs = np.unique(sub, return_counts=True)
print('SUB-SLAB offsets (the ones that matter):', dict(zip(us.tolist(), cs.tolist())))
if a.check:
    bad = sum(n for sh, n in zip(us.tolist(), cs.tolist()) if sh != 0)
    if bad:
        print('\n!! ALIGNMENT CHECK FAILED: %d/%d cases want a non-zero SUB-SLAB shift.'
              % (bad, len(shifts)))
        print('   prep_pairs.py did not land the pairs on the operator grid. Every')
        print('   interpolation baseline and the whole certificate would be measured')
        print('   against a shifted operator. DO NOT continue - re-run stage 1 and')
        print('   check --thin_hint/--thick_hint and the HU range first.')
        sys.exit(1)
    print('\nalignment check PASSED: every case already sits on the operator grid.')
