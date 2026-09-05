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
ap.add_argument('--csv', default=None)
a = ap.parse_args()
r = a.downsample

files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
rows, keep = [], []
for i, f in enumerate(files, 1):
    uid = f[:-9]
    thin = np.load(os.path.join(a.root, f), mmap_mode='r')
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick')), mmap_mode='r')
    nk = thick.shape[0]
    cs = []
    for k in np.linspace(nk * 0.25, nk * 0.75, a.probes).astype(int):
        tk = np.asarray(thick[k], np.float32)
        tn = np.asarray(thin[k * r:(k + 1) * r], np.float32).mean(0)
        if tn.shape != tk.shape or tn.std() < 1e-6 or tk.std() < 1e-6:
            continue
        cs.append(float(np.corrcoef(tn.ravel(), tk.ravel())[0, 1]))
    c = float(np.median(cs)) if cs else 0.0
    ok = c >= a.min_corr
    rows.append(dict(case=uid, inplane_corr=round(c, 4), keep=int(ok)))
    if ok: keep.append(f)
    if i % 25 == 0 or i == len(files):
        print(f'  {i}/{len(files)}  保留 {len(keep)}', flush=True)

c = np.array([x['inplane_corr'] for x in rows])
print(f'\n{len(files)} 例，保留 {len(keep)}（{100*len(keep)/max(len(files),1):.0f}%）')
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
                if not os.path.exists(dst): os.symlink(os.path.abspath(src), dst)
    ap_src = os.path.join(a.root, 'pair_audit.csv')
    if os.path.exists(ap_src): shutil.copy(ap_src, a.out)
    print(f'-> {a.out}')
out = a.csv or os.path.join(a.root, 'pair_screen.csv')
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['case', 'inplane_corr', 'keep']); w.writeheader(); w.writerows(rows)
print('wrote', out)
print('\n排除的病例不是坏数据 —— 它们是同一患者的两次独立采集，')
print('对这个方法不适用，但这个比例本身是要报告的发现（CLAIM 数据流图）。')
