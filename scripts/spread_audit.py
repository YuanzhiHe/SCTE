"""Does sample-to-sample spread flag the fabricated foci?

The block-wise certificate failed for a structural reason: delta and the
data-consistency residual both test agreement with the thick observation, and a
fabrication that averages correctly through-plane satisfies both by construction.
Sample spread does not have that defect - it asks whether the sampler keeps drawing
the same structure, which is a question about the posterior, not about the data term.

A useful flag must concentrate: the fabricated foci should sit in the high-spread
tail. Random performance means the flag carries no information.
"""
import argparse, os, sys, numpy as np
from scipy import ndimage
ap = argparse.ArgumentParser()
ap.add_argument('--root', default='DATA/aligned_test')
ap.add_argument('--recon_dir', default='RECON/spread')
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--thr', type=float, default=-400.)
ap.add_argument('--min_vox', type=int, default=8); ap.add_argument('--max_vox', type=int, default=1000)
a = ap.parse_args()

def foci(vol, lung):
    m=(vol>a.thr)&lung; lab,n=ndimage.label(m)
    if n==0: return np.zeros_like(lab),[]
    sz=ndimage.sum(m,lab,range(1,n+1)); keep=np.nonzero((sz>=a.min_vox)&(sz<=a.max_vox))[0]+1
    return np.where(np.isin(lab,keep),lab,0), keep.tolist()

files=sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
S_fab, S_real = [], []
for f in files:
    rp=os.path.join(a.recon_dir,f.replace('_thin','_rec'))
    sp=os.path.join(a.recon_dir,f.replace('_thin','_std'))
    mp=os.path.join(a.root,f.replace('_thin','_lung'))
    if not all(os.path.exists(x) for x in (rp,sp,mp)): continue
    thin=np.load(os.path.join(a.root,f)).astype(np.float32)
    rec=np.load(rp).astype(np.float32); std=np.load(sp).astype(np.float32)
    lung=np.load(mp)>0
    ref_dense=(thin>a.thr)&lung
    tl,ti=foci(rec,lung)
    for i in ti:
        m=tl==i
        (S_fab if not ref_dense[m].any() else S_real).append(float(std[m].mean()))
    print(f'{f[:-9]}: 虚构 {len(S_fab)} 真实 {len(S_real)}',flush=True)
fab=np.array(S_fab); real=np.array(S_real)
print(f'\n虚构灶 {len(fab)} 个，真实灶 {len(real)} 个')
print(f'灶内平均采样离散度: 虚构 {fab.mean():.2f} HU  真实 {real.mean():.2f} HU  '
      f'比值 {fab.mean()/max(real.mean(),1e-9):.2f}x')
allv=np.concatenate([fab,real]); lab=np.concatenate([np.ones(len(fab)),np.zeros(len(real))])
print(f"\n{'按离散度取最高':>14s}{'捕获虚构灶':>12s}{'随机基线':>10s}")
for q in (1,5,10,20):
    t=np.percentile(allv,100-q); k=allv>=t
    print(f"{q:13d}%{100*lab[k].sum()/max(lab.sum(),1):11.1f}%{q:9d}%")
order=np.argsort(-allv); cum=np.cumsum(lab[order])/max(lab.sum(),1)
auc=float(np.trapezoid(cum,dx=1/len(order)))
print(f"\n排序后的富集 AUC = {auc:.3f}（0.5 = 与随机无异，1.0 = 完美排序）")
