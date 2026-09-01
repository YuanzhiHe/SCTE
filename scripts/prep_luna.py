"""Turn LUNA16 into thin/thick pairs with real nodule annotations.

Two things this can and cannot answer, stated up front because they bound every number
that comes out of it:

  CAN  - does the reconstruction recover REAL, radiologist-annotated nodules that a
         5 mm series loses? Every detection result so far counted connected components,
         which are overwhelmingly vessel cross-sections; this is the first lesion-level
         ground truth in the project.
  CANNOT - say anything about REAL 5 mm reconstructions. LUNA16 ships thin scans only,
         so the thick series here is SIMULATED with our own forward operator. That is
         the pseudo-LR setting the RPLHR authors built their dataset to escape, and a
         method can look good on simulated degradation and fail on scanner output. The
         private cohorts, with genuine 5 mm/1 mm pairs, remain the real test.

Only scans whose native z spacing is <= --max_z are kept: a 2.5 mm LUNA16 scan cannot
stand in for a 1 mm reference.

  python scripts/prep_luna.py --src DATA/luna16 --out DATA/luna_pairs --limit 40
"""
import argparse, csv, os, sys
import numpy as np, torch, SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='DATA/luna16'); ap.add_argument('--out', required=True)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--slice_fwhm', type=float, default=6.25)
ap.add_argument('--max_z', type=float, default=1.25, help='max native z spacing (mm)')
ap.add_argument('--target_z', type=float, default=1.0,
                help='resample every scan to this z spacing before simulating the thick '
                     'series. LUNA16 z spacing runs 0.5-2.5 mm, so a fixed 5x decimation '
                     'would produce a 2.5 mm "thick" slice for one scan and 6.25 mm for '
                     'another - different tasks scored as one.')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
op = SSPForwardOperator(slice_fwhm_mm=a.slice_fwhm, downsample=a.downsample)

ann = {}
for r in csv.DictReader(open(os.path.join(a.src, 'annotations.csv'))):
    ann.setdefault(r['seriesuid'], []).append(
        (float(r['coordX']), float(r['coordY']), float(r['coordZ']), float(r['diameter_mm'])))

mhds = []
for d in sorted(os.listdir(a.src)):
    p = os.path.join(a.src, d)
    if os.path.isdir(p) and d.startswith('subset'):
        mhds += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith('.mhd')]
print(f'{len(mhds)} scans on disk, {len(ann)} annotated series')

rows, kept = [], 0
for p in mhds:
    uid = os.path.basename(p)[:-4]
    if uid not in ann:
        continue
    img = sitk.ReadImage(p)
    sx, sy, sz = img.GetSpacing()
    if sz > a.max_z:
        continue
    if abs(sz - a.target_z) > 1e-3:                            # unify the z grid first
        n_out = int(round(img.GetSize()[2] * sz / a.target_z))
        rs = sitk.ResampleImageFilter()
        rs.SetOutputSpacing((sx, sy, a.target_z))
        rs.SetSize((img.GetSize()[0], img.GetSize()[1], n_out))
        rs.SetOutputOrigin(img.GetOrigin()); rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear); rs.SetDefaultPixelValue(-1024)
        img = rs.Execute(img)
        sz = a.target_z
    vol = sitk.GetArrayFromImage(img).astype(np.float32)      # (z, y, x), already HU
    vol = np.clip(vol, -1024, 1024)
    D = (vol.shape[0] // a.downsample) * a.downsample
    vol = vol[:D]
    thick = op(torch.from_numpy(vol[None, None] / 1000.))[0, 0].numpy() * 1000.
    np.save(os.path.join(a.out, f'{uid}_thin.npy'), vol.astype(np.float32))
    np.save(os.path.join(a.out, f'{uid}_thick.npy'), thick.astype(np.float32))
    ox, oy, oz = img.GetOrigin()
    for (cx, cy, cz, dia) in ann[uid]:
        # world -> voxel; LUNA16 stores world coordinates, and the volume was trimmed
        # to a multiple of r at the END, so the origin is unchanged.
        iz, iy, ix = (cz - oz) / sz, (cy - oy) / sy, (cx - ox) / sx   # sz is post-resample
        if not (0 <= iz < D):
            continue
        rows.append(dict(seriesuid=uid, z=round(iz, 2), y=round(iy, 2), x=round(ix, 2),
                         diameter_mm=round(dia, 2), sx=round(sx, 4), sy=round(sy, 4),
                         sz=round(sz, 4)))
    kept += 1
    print(f'[{kept}] {uid[-12:]}  z={sz:.2f}mm  thin {vol.shape} -> thick {thick.shape}  '
          f'{len(ann[uid])} nodules', flush=True)
    if a.limit and kept >= a.limit:
        break

with open(os.path.join(a.out, 'nodules.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'\n{kept} scans, {len(rows)} nodules -> {a.out}')
