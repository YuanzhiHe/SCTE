"""Generate TotalSegmentator lung-lobe masks for prepared pairs.

Why this exists (measured, not stylistic). The HU-window mask this project used
until now is NOT the clinical lung: on one real case it gives LAA-950 = 8.91%
against 1.05% with the anatomical lobe mask, because its denominator is the
(-990, -500) HU band rather than the parenchyma. Relative comparisons between
models survive that error; absolute densitometry does not.

Writes <case>_lung.npy (uint8, 0 = background, 1..5 = the five lobes) next to
<case>_thin.npy, so every downstream metric can use the same mask.

  python scripts/make_lung_masks.py --root DATA/public_pairs_test [--fast] [--limit N]
"""
import argparse, os, shutil, subprocess, sys, tempfile
import nibabel as nib
import numpy as np

LOBES = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
         'lung_middle_lobe_right', 'lung_lower_lobe_right']
# Windows: the console-script .exe breaks totalsegmentator's multiprocessing named
# pipes when the venv lives on another drive (E:) than the base python (C:).
# Invoking the module through the interpreter keeps the process tree consistent.
TS = [sys.executable, '-m', 'totalsegmentator.bin.TotalSegmentator']


def segment(vol_hu, fast=True, spacing=(1., 1., 1.)):
    tmp = tempfile.mkdtemp()
    # Uncompressed .nii on purpose: temp sits on SSD, and zlib-compressing ~100 MB
    # of float32 per case costs 10-30 s of single-threaded CPU for nothing there.
    nii = os.path.join(tmp, 'in.nii'); out = os.path.join(tmp, 'seg')
    nib.save(nib.Nifti1Image(np.transpose(vol_hu, (2, 1, 0)).astype(np.float32),
                             np.diag(list(spacing) + [1.])), nii)
    cmd = TS + ['-i', nii, '-o', out, '--roi_subset'] + LOBES + (['--fast'] if fast else [])
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    m = np.zeros(vol_hu.shape, np.uint8)
    for i, n in enumerate(LOBES, 1):
        p = os.path.join(out, n + '.nii.gz')
        if os.path.exists(p):
            m[np.transpose(nib.load(p).get_fdata(), (2, 1, 0)) > 0.5] = i
    shutil.rmtree(tmp, ignore_errors=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--suffix', default='_thin.npy')
    ap.add_argument('--fast', action='store_true', default=True)
    ap.add_argument('--full', action='store_true', help='full-resolution model (slower)')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.root) if f.endswith(a.suffix))
    if a.limit:
        files = files[:a.limit]
    ok = skip = 0
    for i, f in enumerate(files, 1):
        outp = os.path.join(a.root, f.replace(a.suffix, '_lung.npy'))
        if os.path.exists(outp):
            skip += 1; continue
        v = np.load(os.path.join(a.root, f)).astype(np.float32)
        try:
            m = segment(v, fast=not a.full)
        except Exception as e:
            print('[skip] %s: %s' % (f, e)); skip += 1; continue
        np.save(outp, m)
        lung = m > 0
        hu = (v > -990) & (v < -500)
        print('[%3d/%d] %-26s lung %5.1f%% of volume | LAA-950 TS %6.3f%% vs HU-window %6.3f%%'
              % (i, len(files), f.replace(a.suffix, ''), 100 * lung.mean(),
                 100 * (v[lung] < -950).mean() if lung.any() else float('nan'),
                 100 * (v[hu] < -950).mean() if hu.any() else float('nan')), flush=True)
        ok += 1
    print('done: %d written, %d skipped/existing -> %s' % (ok, skip, a.root))


if __name__ == '__main__':
    main()
