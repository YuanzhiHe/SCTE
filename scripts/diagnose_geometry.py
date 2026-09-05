"""Read the in-plane geometry straight from the source DICOM.

Two hypotheses have now been wrong - separate acquisitions (acquisition times match) and
a sub-slab z offset (realignment moved the correlation 0.726 -> 0.794 and changed nothing
else). Both were inferred from the processed arrays. This reads the fields that decide
the question instead of inferring them:

  PixelSpacing, Rows/Columns   - same FOV with a different matrix means different mm per
                                 voxel, and prep_pairs crops both volumes with the SAME
                                 index range, so the two crops then cover different
                                 anatomy at different scales. No shift or z offset fixes
                                 that, which is exactly what we observe.
  ImagePositionPatient         - a different reconstruction centre at equal FOV
  ReconstructionDiameter       - already known to match

  python scripts/diagnose_geometry.py --cases /data/henan --n 10
"""
import argparse, os, sys
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True, help='the ORIGINAL case directory tree')
ap.add_argument('--thin_hint', default='1mm'); ap.add_argument('--thick_hint', default='5mm')
ap.add_argument('--n', type=int, default=10)
a = ap.parse_args()

TAGS = {'0028|0030': 'PixelSpacing', '0028|0010': 'Rows', '0028|0011': 'Columns',
        '0020|0032': 'ImagePosition', '0018|1100': 'ReconDiameter',
        '0018|1210': 'Kernel', '0020|0012': 'AcqNumber', '0008|0032': 'AcqTime'}


def header(folder):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(folder)
    if not ids: return None, 0
    best, n = None, -1
    for sid in ids:
        f = r.GetGDCMSeriesFileNames(folder, sid)
        if len(f) > n: best, n = f, len(f)
    rd = sitk.ImageFileReader(); rd.SetFileName(best[0]); rd.ReadImageInformation()
    return {v: (rd.GetMetaData(k).strip() if rd.HasMetaDataKey(k) else '?')
            for k, v in TAGS.items()}, len(ids)


def find(case_dir, hint):
    for d in sorted(os.listdir(case_dir)):
        p = os.path.join(case_dir, d)
        if os.path.isdir(p) and hint.lower() in d.lower(): return p
    return None


cases = sorted(d for d in os.listdir(a.cases) if os.path.isdir(os.path.join(a.cases, d)))[:a.n]
print(f"{'case':<14s}{'字段':<16s}{'薄层(1mm)':>26s}{'厚层(5mm)':>26s}{'':>4s}")
mismatch = {}
for c in cases:
    cd = os.path.join(a.cases, c)
    tp, kp = find(cd, a.thin_hint), find(cd, a.thick_hint)
    if not (tp and kp): print(f"{c[:14]:<14s}  找不到 1mm/5mm 子目录"); continue
    ht, nt = header(tp); hk, nk = header(kp)
    if not (ht and hk): print(f"{c[:14]:<14s}  读不出序列"); continue
    first = True
    for f in TAGS.values():
        same = ht[f] == hk[f]
        if not same: mismatch[f] = mismatch.get(f, 0) + 1
        if not same or f in ('PixelSpacing', 'Rows'):
            print(f"{(c[:14] if first else ''):<14s}{f:<16s}{ht[f][:24]:>26s}{hk[f][:24]:>26s}"
                  f"{'' if same else '  ✗':>4s}")
            first = False
    if nt > 1 or nk > 1:
        print(f"{'':<14s}{'序列数':<16s}{nt:>26d}{nk:>26d}   ← 目录内不止一个序列")
print(f"\n{len(cases)} 例中各字段不一致的例数：")
for k, v in sorted(mismatch.items(), key=lambda x: -x[1]): print(f"  {k:<20s} {v}")
if not mismatch: print("  无 —— 几何完全一致，问题不在 DICOM 头")
