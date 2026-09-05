"""Read the TRUE z spacing of both series from ImagePositionPatient.

SliceThickness is how thick a slice is; the spacing between slice centres is a separate
field and often absent. Measuring it from consecutive ImagePositionPatient values is the
only reliable route, and it settles which series deviates: a thick series reconstructed
at 6.4 mm, or a "1 mm" thin series reconstructed at 0.78 mm increments. The measured
index ratios (median 6.404, range 5.000-6.772, non-integer) must come from one of them.

  python scripts/read_spacing.py --cases /data/henan --n 10
"""
import argparse, os
import numpy as np, SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True)
ap.add_argument('--thin_hint', default='1mm'); ap.add_argument('--thick_hint', default='5mm')
ap.add_argument('--n', type=int, default=10)
a = ap.parse_args()


def zspacing(folder):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(folder)
    if not ids: return None
    best, n = None, -1
    for sid in ids:
        f = r.GetGDCMSeriesFileNames(folder, sid)
        if len(f) > n: best, n = f, len(f)
    zs, th = [], None
    for fn in best[:40]:
        rd = sitk.ImageFileReader(); rd.SetFileName(fn); rd.ReadImageInformation()
        if rd.HasMetaDataKey('0020|0032'):
            zs.append(float(rd.GetMetaData('0020|0032').split('\\')[2]))
        if th is None and rd.HasMetaDataKey('0018|0050'):
            th = float(rd.GetMetaData('0018|0050'))
    zs = np.sort(np.array(zs))
    d = np.diff(zs)
    return dict(n=n, spacing=float(np.median(d)) if len(d) else float('nan'),
                spacing_sd=float(np.std(d)) if len(d) else float('nan'), thickness=th)


def find(cd, hint):
    for d in sorted(os.listdir(cd)):
        p = os.path.join(cd, d)
        if os.path.isdir(p) and hint.lower() in d.lower(): return p
    return None


cases = sorted(d for d in os.listdir(a.cases) if os.path.isdir(os.path.join(a.cases, d)))[:a.n]
print(f"{'case':<14s}{'薄层厚/间距':>16s}{'厚层厚/间距':>16s}{'间距比':>9s}{'间距抖动':>10s}")
ratios = []
for c in cases:
    cd = os.path.join(a.cases, c)
    tp, kp = find(cd, a.thin_hint), find(cd, a.thick_hint)
    if not (tp and kp): print(f"{c[:14]:<14s}  找不到子目录"); continue
    t, k = zspacing(tp), zspacing(kp)
    if not (t and k): print(f"{c[:14]:<14s}  读不出"); continue
    ratio = k['spacing'] / max(t['spacing'], 1e-6)
    ratios.append(ratio)
    ts = '%s/%.3f' % (t['thickness'], t['spacing'])
    ks = '%s/%.3f' % (k['thickness'], k['spacing'])
    jitter = max(t['spacing_sd'], k['spacing_sd'])
    print(f"{c[:14]:<14s}{ts:>16s}{ks:>16s}{ratio:>9.3f}{jitter:>10.4f}")
if ratios:
    r = np.array(ratios)
    print(f"\n间距比：中位 {np.median(r):.3f}  范围 [{r.min():.3f}, {r.max():.3f}]")
    print("这个比值应当与 measure_zmap.py 实测的索引斜率一致。")
    print("薄层间距 < 层厚 = 重叠重建；厚层间距 > 层厚 = 有间隙重建。")
