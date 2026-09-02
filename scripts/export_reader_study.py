"""Export blinded three-arm reader-study material.

Three arms, not two. With only 5 mm and the reconstruction, a nodule a reader reports in
the reconstruction cannot be classified: "the 5 mm series missed it" and "the
reconstruction invented it" have opposite clinical meanings and identical appearance.
The real 1 mm series is what separates them, so it is a required arm.

Arm labels are randomised into the filenames and the key stays behind on the hospital
machine. Images never leave; only the completed score sheet does.

  python scripts/export_reader_study.py --root PRIVATE/henan/pairs/test \
      --recon RECON/henan --out READER --n 130
"""
import argparse, csv, hashlib, os, sys
import numpy as np, SimpleITK as sitk, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--recon', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--n', type=int, default=0)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--seed', type=int, default=20260901)
ap.add_argument('--spacing', type=float, nargs=2, default=[0.7, 0.7])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
op = SSPForwardOperator(slice_fwhm_mm=5.0, downsample=a.downsample)
rng = np.random.default_rng(a.seed)

files = sorted(f for f in os.listdir(a.root) if f.endswith('_thin.npy'))
if a.n: files = files[:a.n]
rows = []
for f in files:
    case = f[:-9]
    rp = os.path.join(a.recon, f.replace('_thin', '_rec'))
    if not os.path.exists(rp):
        print(f'[skip] {case}: no reconstruction'); continue
    thin = np.load(os.path.join(a.root, f)).astype(np.float32)
    thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
    rec = np.load(rp).astype(np.float32)
    # arm A is shown on the 1 mm grid too, so slice count cannot leak the arm
    up = op.upsample_to_grid(torch.from_numpy(thick / 1000.)[None, None],
                             thin.shape[0])[0, 0].numpy() * 1000.
    arms = {'A_thick': up, 'B_recon': rec, 'C_thin': thin}
    ids = rng.permutation(3)
    for (arm, vol), k in zip(arms.items(), ids):
        rid = hashlib.sha1(f'{case}|{arm}|{a.seed}'.encode()).hexdigest()[:10]
        img = sitk.GetImageFromArray(np.clip(vol, -1024, 1024).astype(np.int16))
        img.SetSpacing((a.spacing[0], a.spacing[1], 1.0))
        sitk.WriteImage(img, os.path.join(a.out, f'{rid}.nii.gz'))
        rows.append(dict(reader_id=rid, case=case, arm=arm, order=int(k)))
    print(f'[ok] {case}', flush=True)

with open(os.path.join(a.out, 'reader_manifest_DO_NOT_EXPORT.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['reader_id', 'case', 'arm', 'order'])
    w.writeheader(); w.writerows(rows)
with open(os.path.join(a.out, 'reader_blank.csv'), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['reader_id', 'reader_name', 'nodule_found', 'slice', 'lobe',
                'diameter_mm', 'type_solid_partsolid_ggo', 'confidence_1_5',
                'image_quality_1_5', 'suspected_artefact', 'notes'])
    for r in sorted(rows, key=lambda x: x['reader_id']):
        w.writerow([r['reader_id']] + [''] * 10)
print(f'\n{len(rows)//3} cases x 3 arms -> {a.out}')
print('reader_manifest_DO_NOT_EXPORT.csv 保留在院内；只有填好的 reader_blank.csv 可带出')
