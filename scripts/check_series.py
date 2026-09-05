"""Why does the DICOM say ratio 5 while the arrays measure 6.4?

read_spacing takes the MEDIAN gap over the first 40 files, which hides duplicates,
gaps and non-uniform sampling. If the array has more slices than its z extent divided by
the nominal spacing, the series contains slices the nominal spacing does not account for
- duplicated positions, two interleaved reconstructions, or a variable increment - and
that inflates the index ratio without changing the median gap.

Reports, per series: slice count, z extent, the spacing that count and extent IMPLY, and
the full histogram of consecutive gaps.

  python scripts/check_series.py --cases /data/henan --n 6
"""
import argparse, os
import numpy as np, SimpleITK as sitk

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True)
ap.add_argument('--thin_hint', default='1mm'); ap.add_argument('--thick_hint', default='5mm')
ap.add_argument('--n', type=int, default=6)
a = ap.parse_args()


def inspect(folder):
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(folder)
    if not ids: return None
    out = []
    for sid in ids:
        files = r.GetGDCMSeriesFileNames(folder, sid)
        zs = []
        for fn in files:
            rd = sitk.ImageFileReader(); rd.SetFileName(fn); rd.LoadPrivateTagsOff()
            rd.ReadImageInformation()
            if rd.HasMetaDataKey('0020|0032'):
                zs.append(float(rd.GetMetaData('0020|0032').split('\\')[2]))
        if len(zs) < 3: continue
        z = np.sort(np.array(zs))
        d = np.diff(z)
        ext = float(z[-1] - z[0])
        out.append(dict(sid=sid[-8:], n=len(files), n_pos=len(zs), extent=ext,
                        implied=ext / max(len(zs) - 1, 1), med=float(np.median(d)),
                        dup=int((d < 1e-3).sum()), gaps=d))
    return out


def find(cd, hint):
    for d in sorted(os.listdir(cd)):
        p = os.path.join(cd, d)
        if os.path.isdir(p) and hint.lower() in d.lower(): return p
    return None


cases = sorted(d for d in os.listdir(a.cases) if os.path.isdir(os.path.join(a.cases, d)))[:a.n]
for c in cases:
    cd = os.path.join(a.cases, c)
    print(f"\n=== {c[:20]}")
    info = {}
    for role, hint in (('薄层', a.thin_hint), ('厚层', a.thick_hint)):
        p = find(cd, hint)
        if not p: print(f"  {role}: 找不到目录"); continue
        ss = inspect(p)
        if not ss: print(f"  {role}: 读不出"); continue
        big = max(ss, key=lambda x: x['n'])
        info[role] = big
        print(f"  {role} 序列数 {len(ss)}  选中 …{big['sid']}  层数 {big['n']}  "
              f"z跨度 {big['extent']:.1f}mm")
        print(f"       中位间距 {big['med']:.3f}  由层数/跨度反推 {big['implied']:.3f}"
              f"  重复位置 {big['dup']}")
        u, n = np.unique(np.round(big['gaps'], 2), return_counts=True)
        top = sorted(zip(u.tolist(), n.tolist()), key=lambda x: -x[1])[:4]
        print(f"       间距分布 {top}")
        if len(ss) > 1:
            print(f"       !! 目录内有 {len(ss)} 个序列，层数分别 "
                  f"{sorted((x['n'] for x in ss), reverse=True)}")
    if '薄层' in info and '厚层' in info:
        t, k = info['薄层'], info['厚层']
        print(f"  -> 层数比 {t['n']/max(k['n'],1):.3f}   "
              f"中位间距比 {k['med']/max(t['med'],1e-6):.3f}   "
              f"反推间距比 {k['implied']/max(t['implied'],1e-6):.3f}")
        if abs(t['n']/max(k['n'],1) - k['med']/max(t['med'],1e-6)) > 0.3:
            print("     !! 层数比与间距比不一致 —— 序列里有间距无法解释的层")
