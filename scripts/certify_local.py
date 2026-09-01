"""Block-wise certificate: WHERE is the reconstruction untrustworthy?

The per-scan certificate reports one displacement for a whole lung. That is the right
granularity for a densitometry claim - the reported LAA is also one number - but it is
the wrong granularity for the deployment target. A township hospital reads images, and
a few hundred fabricated voxels can sit inside a scan whose global mean is perfect.

So: tile the volume, run the same estimators per tile, and emit a per-tile verdict.

The point of this script is not the map, it is the VALIDATION at the end. A safety
flag that does not concentrate on the places where things actually went wrong is
decoration. So each tile is scored against the ground truth twice - by its certificate
quantities (reference-free) and by its actual fabricated/erased lesion count (needs the
1 mm reference) - and the script reports whether the first predicts the second.

  python scripts/certify_local.py --recon_dir RECON/s1 --n 8
"""
import argparse, os, sys
import numpy as np, torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator
from scte_r.agentic import lung_set, eroded_lung_set

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='DATA/aligned_test')
ap.add_argument('--recon_dir', required=True)
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--tile', type=int, nargs=3, default=[40, 64, 64])
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--slice_fwhm', type=float, default=5.0)
ap.add_argument('--thr', type=float, default=-400.)
ap.add_argument('--min_vox', type=int, default=8)
ap.add_argument('--max_vox', type=int, default=1000)
ap.add_argument('--csv', default=None)
a = ap.parse_args()
r = a.downsample
op = SSPForwardOperator(slice_fwhm_mm=a.slice_fwhm, downsample=r)


def tile_stats(rec_t, thick_t):
    """Reference-free quantities for one tile.

    delta: the mean HU the reconstruction sits away from what the thick observation
      implies, using the operator's slab-mean preservation. This is the local version
      of the per-scan estimator.
    dc:    RMS data-consistency residual, ||A(x_hat) - y|| / ||y - mean(y)||.
    """
    x = torch.from_numpy(rec_t[None, None] / 1000.).float()
    y = torch.from_numpy(thick_t[None, None] / 1000.).float()
    ax = op(x)
    n = min(ax.shape[2], y.shape[2])
    if n < 2:
        return float('nan'), float('nan')
    ax, y = ax[:, :, :n], y[:, :, :n]
    delta = float((ax - y).mean()) * 1000.
    denom = float((y - y.mean()).pow(2).mean().sqrt())
    dc = float((ax - y).pow(2).mean().sqrt() / max(denom, 1e-8))
    return delta, dc


def foci(vol, lung):
    m = (vol > a.thr) & lung
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(lab), []
    sz = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.nonzero((sz >= a.min_vox) & (sz <= a.max_vox))[0] + 1
    return np.where(np.isin(lab, keep), lab, 0), keep.tolist()


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
td, th, tw = a.tile
rows = []
for f in files:
    rp = os.path.join(a.recon_dir, f.replace('_thin', '_rec'))
    mp = os.path.join(a.root, f.replace('_thin', '_lung'))
    if not (os.path.exists(rp) and os.path.exists(mp)):
        continue
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    rec = np.load(rp).astype(np.float32)
    lung = np.load(mp) > 0
    rl, ri = foci(thin, lung)
    tl, ti = foci(rec, lung)
    ref_dense = (thin > a.thr) & lung
    rec_dense = (rec > a.thr) & lung
    D, H, W = thin.shape
    for z in range(0, D - td + 1, td):
        z -= z % r
        for yy in range(0, H - th + 1, th):
            for xx in range(0, W - tw + 1, tw):
                sl = (slice(z, z + td), slice(yy, yy + th), slice(xx, xx + tw))
                lm = lung[sl]
                if lm.mean() < 0.2:                 # tiles that are barely lung say little
                    continue
                delta, dc = tile_stats(rec[sl], thick[z // r: z // r + td // r, yy:yy + th, xx:xx + tw])
                # ground truth for this tile: foci present in one volume, absent in the other
                fab = sum(1 for i in np.unique(tl[sl]) if i > 0 and
                          not ref_dense[sl][tl[sl] == i].any())
                era = sum(1 for i in np.unique(rl[sl]) if i > 0 and
                          not rec_dense[sl][rl[sl] == i].any())
                rows.append(dict(case=f[:-9], z=z, y=yy, x=xx, lung=float(lm.mean()),
                                 delta=delta, dc=dc, fab=fab, era=era))
    print(f'[{files.index(f)+1}/{len(files)}] {f[:-9]}  tiles so far {len(rows)}', flush=True)

A = lambda k: np.array([q[k] for q in rows], float)
delta, dc, fab, era = np.abs(A('delta')), A('dc'), A('fab'), A('era')
ok = np.isfinite(delta) & np.isfinite(dc)
delta, dc, fab, era = delta[ok], dc[ok], fab[ok], era[ok]
bad = (fab + era) > 0
print(f'\n{len(delta)} tiles, {bad.sum()} ({100*bad.mean():.1f}%) contain a fabricated '
      f'or erased focus\n')
print(f"{'指标':<26s}{'干净块':>10s}{'出问题块':>10s}{'比值':>8s}")
for nm, v in (('|delta| (HU)', delta), ('数据一致性残差', dc)):
    print(f'{nm:<26s}{v[~bad].mean():10.3f}{v[bad].mean():10.3f}'
          f'{v[bad].mean()/max(v[~bad].mean(),1e-9):8.2f}x')

print(f"\n按局部指标取最高的 N% 块，能覆盖多少出问题的块（随机基线 = N%）")
print(f"{'取最高':>8s}{'|delta| 覆盖':>14s}{'DC 覆盖':>12s}{'随机':>8s}")
for q in (5, 10, 20, 30):
    for nm, v, col in (('d', delta, 14), ('c', dc, 12)):
        pass
    kd = delta >= np.percentile(delta, 100 - q)
    kc = dc >= np.percentile(dc, 100 - q)
    print(f"{q:7d}%{100*fab[kd].sum()/max(fab.sum(),1)+0:13.1f}%"
          f"{100*fab[kc].sum()/max(fab.sum(),1):11.1f}%{q:7d}%")
print('\n（覆盖率 = 该分位内的块所含虚构灶数 / 全部虚构灶数；显著高于随机才说明这个标志有用）')
if a.csv:
    import csv as _csv
    with open(a.csv, 'w', newline='') as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('wrote', a.csv)
