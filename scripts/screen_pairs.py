"""Keep only pairs that are two reconstructions of the SAME acquisition.

The method's premise is thick = A(thin) applied to one raw acquisition. A hospital
archive does not guarantee that: the 1 mm and 5 mm series can be separate scans, or the
same scan at a different breath-hold. Then the anatomy matches in z, the protocol
matches exactly (kernel, FOV, kVp), and yet no operator relates the voxels - which is
what a relative residual near 1.0 with an in-plane correlation of 0.65-0.84 means.

Public paired datasets are curated to avoid this. Archives are not, so screening is a
required step rather than a safety net, and how many cases fail is itself a finding.

  python scripts/screen_pairs.py --root PRIVATE/henan/pairs --out PRIVATE/henan/pairs_clean
"""
import argparse, csv, os, shutil, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', default=None, help='symlink the surviving pairs here')
ap.add_argument('--min_corr', type=float, default=0.95,
                help='in-plane correlation between a thick slice and its 5 thin slices. '
                     'Same-acquisition pairs sit above 0.99; separate scans land at 0.65-0.85.')
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--probes', type=int, default=5, help='slices sampled per case')
ap.add_argument('--search_offset', action='store_true', default=True,
                help='try every sub-slab offset before judging. Without this the probe '
                     'assumes thick k <-> thin [5k,5k+5) and a merely MISALIGNED pair - '
                     'same acquisition, wrong window - scores like an unrelated one.')
ap.add_argument('--no_search_offset', dest='search_offset', action='store_false')
ap.add_argument('--csv', default=None)
a = ap.parse_args()
r = a.downsample

def link_or_copy(src, dst):
    """Symlink when the platform allows it, copy when it does not.

    Windows refuses symlinks without Developer Mode or admin rights, and these splits
    are pure views over the prepared pairs - so falling back to a copy costs disk but
    never correctness. Patching this by hand on the analysis machine would be undone by
    the next git pull.
    """
    import os, shutil
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
rows, keep = [], []
for i, f in enumerate(files, 1):
    uid = f[:-9]
    thin = np.load(os.path.join(a.root, f), mmap_mode='r')
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')), mmap_mode='r')
    nk = thick.shape[0]
    offs = range(r) if a.search_offset else (0,)
    per_off = {}
    for off in offs:
        cs = []
        for k in np.linspace(nk * 0.25, nk * 0.75, a.probes).astype(int):
            lo = k * r + off
            if lo + r > thin.shape[0]: continue
            tk = np.asarray(thick[k], np.float32)
            tn = np.asarray(thin[lo:lo + r], np.float32).mean(0)
            if tn.shape != tk.shape or tn.std() < 1e-6 or tk.std() < 1e-6:
                continue
            cs.append(float(np.corrcoef(tn.ravel(), tk.ravel())[0, 1]))
        per_off[off] = float(np.median(cs)) if cs else 0.0
    best = max(per_off, key=per_off.get)
    c, c0 = per_off[best], per_off.get(0, 0.0)
    ok = c >= a.min_corr
    rows.append(dict(case=uid, inplane_corr=round(c, 4), corr_at_0=round(c0, 4),
                     best_offset=best, keep=int(ok)))
    if ok: keep.append(f)
    if i % 25 == 0 or i == len(files):
        print(f'  {i}/{len(files)}  保留 {len(keep)}', flush=True)

c = np.array([x['inplane_corr'] for x in rows])
print(f'\n{len(files)} 例，保留 {len(keep)}（{100*len(keep)/max(len(files),1):.0f}%）')
u, n = np.unique([x['best_offset'] for x in rows], return_counts=True)
print(f'最佳亚 slab 偏移分布：{dict(zip(u.tolist(), n.tolist()))}')
c0 = np.array([x['corr_at_0'] for x in rows])
print(f'仅看偏移 0 的话会保留 {int((c0 >= a.min_corr).sum())} 例 —— 差额就是"错位但同源"的病例')
print(f'面内相关分布：中位 {np.median(c):.3f}  '
      f'>0.99 的 {int((c > .99).sum())}  0.9-0.99 的 {int(((c > .9) & (c <= .99)).sum())}  '
      f'<0.9 的 {int((c <= .9).sum())}')
if a.out and keep:
    os.makedirs(a.out, exist_ok=True)
    for f in keep:
        for suf in ('_thin.npy', '_thick.npy', '_lung.npy', '_base.npy'):
            src = os.path.join(a.root, f.replace('_thin.npy', suf))
            if os.path.exists(src):
                dst = os.path.join(a.out, os.path.basename(src))
                link_or_copy(src, dst)
    ap_src = os.path.join(a.root, 'pair_audit.csv')
    if os.path.exists(ap_src): shutil.copy(ap_src, a.out)
    print(f'-> {a.out}')
out = a.csv or os.path.join(a.root, 'pair_screen.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['case', 'inplane_corr', 'corr_at_0', 'best_offset',
                                       'keep']); w.writeheader(); w.writerows(rows)
print('wrote', out)
print('\n排除的病例不是坏数据 —— 它们是同一患者的两次独立采集，')
print('对这个方法不适用，但这个比例本身是要报告的发现（CLAIM 数据流图）。')
