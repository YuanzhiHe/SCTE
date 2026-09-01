"""FROC-style detection: sensitivity vs false positives per scan.

Lung-nodule CAD does not match by IoU - a detection counts as a hit when its centre
falls inside the reference lesion (LUNA16 convention). That is the right criterion for
a triage claim ("did it point the reader at the lesion"), and it is what reviewers in
this area expect. Morphological agreement is a separate claim and is reported
separately; it should not be silently folded into a detection number.
"""
import argparse, os, numpy as np
from scipy import ndimage
ap=argparse.ArgumentParser()
ap.add_argument('--root',default='DATA/aligned_test'); ap.add_argument('--n',type=int,default=8)
ap.add_argument('--thr',type=float,default=-400.)
ap.add_argument('--min_vox',type=int,default=8); ap.add_argument('--max_vox',type=int,default=1000)
ap.add_argument('--recon_dir',default=None); ap.add_argument('--label',default='flow')
a=ap.parse_args()
import torch, sys; sys.path.insert(0,'/home/prinlab/SCTE')
from scte_r.forward_operator import SSPForwardOperator
op=SSPForwardOperator(slice_fwhm_mm=5.0,downsample=5)
def foci(v,l):
    m=(v>a.thr)&l; lab,n=ndimage.label(m)
    if n==0: return np.zeros_like(lab),[],np.zeros((0,3))
    sz=ndimage.sum(m,lab,range(1,n+1)); k=np.nonzero((sz>=a.min_vox)&(sz<=a.max_vox))[0]+1
    lab2=np.where(np.isin(lab,k),lab,0)
    cen=np.array(ndimage.center_of_mass(m,lab,k)) if len(k) else np.zeros((0,3))
    return lab2,k.tolist(),cen
res={}
files=sorted(x for x in os.listdir(a.root) if x.endswith('_thin.npy'))[:a.n]
for f in files:
    mp=os.path.join(a.root,f.replace('_thin','_lung'))
    if not os.path.exists(mp): continue
    thin=np.load(os.path.join(a.root,f)).astype(np.float32)
    thick=np.load(os.path.join(a.root,f.replace('_thin','_thick'))).astype(np.float32)
    lung=np.load(mp)>0
    up=op.upsample_to_grid(torch.from_numpy(thick/1000.)[None,None],thin.shape[0])[0,0].numpy()*1000
    arms={'上采样(现状)':up}
    bp=os.path.join(a.root,f.replace('_thin','_base'))
    if os.path.exists(bp): arms['CTHNet']=np.load(bp).astype(np.float32)
    if a.recon_dir:
        rp=os.path.join(a.recon_dir,f.replace('_thin','_rec'))
        if os.path.exists(rp): arms[a.label]=np.load(rp).astype(np.float32)
    rl,ri,_=foci(thin,lung)
    for k,v in arms.items():
        tl,ti,cen=foci(v,lung)
        hit=np.zeros(len(ti),bool); found=set()
        for idx,c in enumerate(cen):
            ci=tuple(int(round(x)) for x in c)
            ci=tuple(min(max(ci[d],0),thin.shape[d]-1) for d in range(3))
            lbl=rl[ci]                       # centre inside a reference lesion?
            if lbl>0: hit[idx]=True; found.add(int(lbl))
        r=res.setdefault(k,[0,0,0,0])
        r[0]+=len(ri); r[1]+=len(found); r[2]+=int((~hit).sum()); r[3]+=1
print(f"FROC 式检出（中心点落在参考病灶内即命中），thr={a.thr:.0f} HU，n={len(files)} 例\n")
print(f"{'臂':<16s}{'参考灶':>8s}{'命中':>8s}{'灵敏度':>9s}{'假阳/例':>10s}")
for k,(nref,nhit,nfp,ncase) in res.items():
    print(f"{k:<16s}{nref:8d}{nhit:8d}{100*nhit/max(nref,1):8.1f}%{nfp/max(ncase,1):10.1f}")
