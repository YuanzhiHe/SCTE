"""Does the reconstruction invent focal lesions, or erase them?

This is the safety question for the deployment target: a township hospital with only
thick-slice CT, where nobody can check the output against a 1 mm reference. A blurry
image is an honest "I don't know"; a fabricated nodule is a confident lie that sends a
patient to an unnecessary biopsy, and an erased one is a missed cancer.

Method. Inside the lung, take voxels above `--thr` HU and keep connected components in
a nodule-like size range. A component in the reconstruction with no overlapping
component in the 1 mm reference is a FABRICATION; a reference component with no
overlapping reconstruction component is an ERASURE.

Read the numbers RELATIVELY. Pulmonary vessels are dense and connected, and their
oblique cross-sections look like nodules to any size-and-threshold rule, so the
absolute counts are inflated for every arm alike. What the comparison answers is
whether the generative stage fabricates MORE than the deterministic backbone it sits
on - which is exactly the question a safety case has to answer.

  python scripts/hallucination.py --arms base:AL_vol_cthnet flow:AL_final_s1
"""
import argparse, os, sys
import numpy as np, torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='DATA/aligned_test')
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--thr', type=float, default=-400., help='HU threshold for a solid focus')
ap.add_argument('--min_vox', type=int, default=8)
ap.add_argument('--max_vox', type=int, default=1000)
ap.add_argument('--recon_dir', default=None,
                help='directory of <case>_rec.npy volumes to score (one arm)')
ap.add_argument('--label', default='recon')
ap.add_argument('--iou', type=float, default=0.0,
                help='strict matching: a focus counts as found only if it overlaps a '
                     'reference focus with at least this IoU. 0 = the lenient test '
                     '(any dense voxel at that location), which answers "does it invent '
                     'structure from nothing" but not "does it change a lesion\'s shape".')
ap.add_argument('--sweep', default=None, metavar='T1,T2,...',
                help='repeat the audit at several HU thresholds and print a PR table')
a = ap.parse_args()

op = SSPForwardOperator(slice_fwhm_mm=5.0, downsample=5)


def raw_mask(vol, lung):
    return (vol > a.thr) & lung


def foci(vol, lung):
    m = (vol > a.thr) & lung
    lab, n = ndimage.label(m)
    if n == 0:
        return np.zeros_like(lab), []
    sz = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.nonzero((sz >= a.min_vox) & (sz <= a.max_vox))[0] + 1
    out = np.where(np.isin(lab, keep), lab, 0)
    return out, keep.tolist()


def compare_iou(ref_lab, ref_ids, tst_lab, tst_ids, min_iou):
    """Greedy IoU matching, computed in ONE pass over the volume.

    The obvious implementation compares every reconstruction component against the
    reference volume, which is O(n_components x n_voxels) - at ~17k components that is
    hours. Encoding each (ref_label, tst_label) co-occurrence as a single integer and
    counting them with bincount gives every pairwise intersection in one sweep.
    """
    ref_ids = np.asarray(ref_ids); tst_ids = np.asarray(tst_ids)
    if len(ref_ids) == 0 or len(tst_ids) == 0:
        return len(tst_ids), len(ref_ids)
    nt = int(tst_lab.max()) + 1
    both = (ref_lab > 0) & (tst_lab > 0)
    pair = ref_lab[both].astype(np.int64) * nt + tst_lab[both].astype(np.int64)
    cnt = np.bincount(pair)
    nz = np.nonzero(cnt)[0]
    r_of, t_of, inter = nz // nt, nz % nt, cnt[nz]
    r_size = np.bincount(ref_lab.ravel())
    t_size = np.bincount(tst_lab.ravel())
    iou = inter / (r_size[r_of] + t_size[t_of] - inter)
    keep = iou >= min_iou
    matched_ref = set(r_of[keep].tolist())
    matched_tst = set(t_of[keep].tolist())
    fab = int(sum(1 for j in tst_ids if j not in matched_tst))
    era = int(sum(1 for i in ref_ids if i not in matched_ref))
    return fab, era


def compare(ref_lab, ref_ids, tst_lab, tst_ids, ref_raw=None, tst_raw=None):
    """Matched = the focus overlaps dense tissue in the other volume.

    The test is against the RAW supra-threshold mask, not against the other volume's
    size-filtered components. Filtering both sides independently was wrong: one real
    structure can be 1200 voxels in the reference (dropped as too large) and 900 in a
    smoothed reconstruction (kept), and then gets scored as a fabrication AND an
    erasure. That artefact is what made up-sampling look 21.5% fabricating - it
    measures component-topology instability, not invention.
    """
    R = ref_raw if ref_raw is not None else (ref_lab > 0)
    T = tst_raw if tst_raw is not None else (tst_lab > 0)
    fab = sum(1 for i in tst_ids if not R[tst_lab == i].any())
    era = sum(1 for i in ref_ids if not T[ref_lab == i].any())
    return fab, era


files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))[:a.n]
rows = []
print(f"threshold {a.thr:.0f} HU, focus size {a.min_vox}-{a.max_vox} voxels\n")
print(f"{'case':<14s}{'真实灶':>7s}{'  |':>3s}"
      f"{'上采样 造/漏':>16s}{'  |':>3s}{'骨干 造/漏':>15s}{'  |':>3s}{'流匹配 造/漏':>16s}")
for f in files:
    mp = os.path.join(a.root, f.replace('_thin', '_lung'))
    if not os.path.exists(mp):
        continue
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    base_p = os.path.join(a.root, f.replace('_thin', '_base'))
    lung = np.load(mp) > 0
    up = op.upsample_to_grid(torch.from_numpy(thick / 1000.)[None, None],
                             thin.shape[0])[0, 0].numpy() * 1000.
    arms = {'up': up}
    if os.path.exists(base_p):
        arms['base'] = np.load(base_p).astype(np.float32)
    if a.recon_dir:
        rp = os.path.join(a.recon_dir, f.replace('_thin', '_rec'))
        if os.path.exists(rp):
            arms[a.label] = np.load(rp).astype(np.float32)
    rl, ri = foci(thin, lung)
    rraw = raw_mask(thin, lung)
    r = {'case': f[:-9], 'n_ref': len(ri)}
    for k, v in arms.items():
        tl, ti = foci(v, lung)
        fab, era = (compare_iou(rl, ri, tl, ti, a.iou) if a.iou > 0
                    else compare(rl, ri, tl, ti, rraw, raw_mask(v, lung)))
        r[k] = (fab, era, len(ti))
    rows.append(r)
    g = lambda k: (f"{r[k][0]:5d}/{r[k][1]:<5d}" if k in r else f"{'-':>11s}")
    print(f"{r['case']:<14s}{r['n_ref']:7d}{'  |':>3s}{g('up'):>16s}{'  |':>3s}"
          f"{g('base'):>15s}{'  |':>3s}{g(a.label):>16s}")

print()
n_ref = sum(r['n_ref'] for r in rows)
print(f"参考病灶总数 {n_ref}（n={len(rows)} 例）\n")
print(f"{'臂':<14s}{'检出灶':>8s}{'虚构':>8s}{'漏掉':>8s}{'虚构率':>9s}{'漏检率':>9s}")
for k in ['up', 'base', a.label]:
    if not any(k in r for r in rows):
        continue
    fab = sum(r[k][0] for r in rows if k in r)
    era = sum(r[k][1] for r in rows if k in r)
    tot = sum(r[k][2] for r in rows if k in r)
    nm = {'up': '上采样(现状)', 'base': '骨干(确定性)'}.get(k, k)
    print(f"{nm:<14s}{tot:8d}{fab:8d}{era:8d}{100*fab/max(tot,1):8.1f}%{100*era/max(n_ref,1):8.1f}%")
print("\n虚构率 = 重建里出现但真实 1 mm 里没有的灶 / 重建检出的灶")
print("漏检率 = 真实 1 mm 里有但重建里没有的灶 / 真实灶总数")
