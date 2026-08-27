"""Run TotalSegmentator on one prepared thin volume and report the lung mask."""
import numpy as np, nibabel as nib, os, sys, subprocess, tempfile
case = sys.argv[1] if len(sys.argv) > 1 else 'DATA/public_pairs_test/CT00000102_thin.npy'
v = np.load(case).astype(np.float32)
tmp = tempfile.mkdtemp()
nii = os.path.join(tmp, 'in.nii.gz')
# prepared volumes are (D,H,W) HU at 1mm isotropic-ish; nibabel wants (X,Y,Z)
nib.save(nib.Nifti1Image(np.transpose(v, (2, 1, 0)), np.diag([1., 1., 1., 1.])), nii)
out = os.path.join(tmp, 'seg')
cmd = ['/home/prinlab/miniconda3/envs/scte/bin/TotalSegmentator', '-i', nii, '-o', out, '--fast',
       '--roi_subset', 'lung_upper_lobe_left', 'lung_lower_lobe_left',
       'lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right']
print(' '.join(cmd)); r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout[-1500:]); print(r.stderr[-1500:] if r.returncode else '')
if r.returncode == 0:
    lobes = {}
    for i, n in enumerate(['lung_upper_lobe_left','lung_lower_lobe_left','lung_upper_lobe_right',
                           'lung_middle_lobe_right','lung_lower_lobe_right'], 1):
        p = os.path.join(out, n + '.nii.gz')
        if os.path.exists(p):
            lobes[n] = np.transpose(nib.load(p).get_fdata(), (2, 1, 0)) > 0.5
    m = np.zeros(v.shape, np.uint8)
    for i, (n, a) in enumerate(lobes.items(), 1): m[a] = i
    hu = (v > -990) & (v < -500)
    print("\nvolume %s" % (v.shape,))
    print("TotalSegmentator lung: %.1f%% of voxels, %d lobes" % (100*(m>0).mean(), len(lobes)))
    print("HU-window mask       : %.1f%% of voxels" % (100*hu.mean()))
    print("overlap (TS ∩ HU)/TS : %.3f   |  (TS ∩ HU)/HU: %.3f" % (
        (m>0)[hu].mean() if hu.any() else 0, hu[(m>0)].mean() if (m>0).any() else 0))
    for i,(n,a) in enumerate(lobes.items(),1):
        laa_ts = 100*((v[a] < -950).mean()) if a.any() else float('nan')
        print("  %-26s %6.2f%% of lung   LAA-950 = %5.2f%%" % (n, 100*a.mean()/max((m>0).mean(),1e-9), laa_ts))
    print("\nLAA-950  TotalSegmentator mask: %.3f%%   HU-window mask: %.3f%%" % (
        100*(v[m>0] < -950).mean(), 100*(v[hu] < -950).mean()))
