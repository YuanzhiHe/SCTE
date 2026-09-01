"""Nodule-level FROC on LUNA16: does reconstruction recover lesions a 5 mm series loses?

Every detection number in this project so far counted connected components, which are
overwhelmingly vessel cross-sections. Here the ground truth is 1186 nodules annotated by
radiologists, so "sensitivity" finally means what the word implies.

Detection follows the lung-nodule CAD convention: a candidate counts as a hit when its
centre falls within the annotated nodule's radius. The candidate generator is
deliberately crude and identical across arms - it is a probe of the image, not a CAD
system, and any cleverness in it would confound the comparison.

  python scripts/froc_luna.py --root DATA/luna_pairs
"""
import argparse, csv, os, sys
import numpy as np, torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='DATA/luna_pairs')
ap.add_argument('--n', type=int, default=0)
ap.add_argument('--thr', type=float, default=-400.)
ap.add_argument('--min_vox', type=int, default=8)
ap.add_argument('--max_vox', type=int, default=20000)
ap.add_argument('--recon_dir', default=None); ap.add_argument('--label', default='ours')
ap.add_argument('--downsample', type=int, default=5)
a = ap.parse_args()
op = SSPForwardOperator(slice_fwhm_mm=5.0, downsample=a.downsample)

nod = {}
for r in csv.DictReader(open(os.path.join(a.root, 'nodules.csv'))):
    nod.setdefault(r['seriesuid'], []).append(
        (float(r['z']), float(r['y']), float(r['x']), float(r['diameter_mm']),
         float(r['sy']), float(r['sx'])))


def candidates(vol, lung):
    m = (vol > a.thr) & lung
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros((0, 3))
    sz = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.nonzero((sz >= a.min_vox) & (sz <= a.max_vox))[0] + 1
    if not len(keep):
        return np.zeros((0, 3))
    return np.array(ndimage.center_of_mass(m, lab, keep))


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.n:
    files = files[:a.n]
res = {}
for f in files:
    uid = f[:-9]
    if uid not in nod:
        continue
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    # a crude but arm-independent lung region, computed on each arm's own image
    up = op.upsample_to_grid(torch.from_numpy(thick / 1000.)[None, None],
                             thin.shape[0])[0, 0].numpy() * 1000.
    arms = {'5mm 上采样(现状)': up, '真实1mm(上界)': thin}
    if a.recon_dir:
        rp = os.path.join(a.recon_dir, f.replace('_thin', '_rec'))
        if os.path.exists(rp):
            arms[a.label] = np.load(rp).astype(np.float32)
    for k, v in arms.items():
        lung = ndimage.binary_fill_holes((v > -1000) & (v < -400))
        cen = candidates(v, lung)
        hit = np.zeros(len(nod[uid]), bool)
        used = np.zeros(len(cen), bool)
        for i, (nz, ny, nx, dia, sy, sx) in enumerate(nod[uid]):
            if not len(cen):
                continue
            d = np.sqrt(((cen[:, 0] - nz) * 1.0) ** 2 + ((cen[:, 1] - ny) * sy) ** 2
                        + ((cen[:, 2] - nx) * sx) ** 2)          # mm, z spacing is 1
            j = int(np.argmin(d))
            if d[j] <= max(dia / 2, 1.5):
                hit[i] = True; used[j] = True
        r = res.setdefault(k, [0, 0, 0, 0])
        r[0] += len(hit); r[1] += int(hit.sum()); r[2] += int((~used).sum()); r[3] += 1

print(f'\nLUNA16 结节级 FROC（中心距 <= 半径即命中），thr={a.thr:.0f} HU，'
      f'{res[list(res)[0]][3]} 例，{res[list(res)[0]][0]} 个标注结节\n')
print(f"{'臂':<20s}{'命中':>7s}{'灵敏度':>9s}{'假阳/例':>10s}")
for k, (n, h, fp, nc) in res.items():
    print(f'{k:<20s}{h:7d}{100*h/max(n,1):8.1f}%{fp/max(nc,1):10.1f}')
