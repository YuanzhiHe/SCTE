"""Preprocess public thin-slice CT into the .npy format SimulatedThickDataset reads.

SimulatedThickDataset expects, under --out:
    <case>.npy   float32, HU, shape (D, H, W), z-ordered (head->foot or consistent)

This script reads common public formats and writes exactly that:
  - NIfTI            .nii / .nii.gz            (RPLHR-CT, many public sets)
  - MetaImage        .mhd (+ .raw)            (LUNA16)
  - DICOM series     a folder of .dcm slices  (LIDC-IDRI)

All reading goes through SimpleITK, which applies the CT rescale slope/intercept
so the output is already in HU. Non-CT / tiny volumes are skipped with a warning.

Usage
-----
  pip install SimpleITK
  # NIfTI or .mhd files sitting in a folder:
  python scripts/prep_public.py --in /data/RPLHR/thin --out DATA/public_thin
  # DICOM: each case is its own sub-folder of .dcm under --in:
  python scripts/prep_public.py --in /data/LIDC --out DATA/public_thin --dicom

Notes
-----
* Only thin-slice volumes are needed here; the thick series is SIMULATED on the
  fly by SimulatedThickDataset via the SSP forward operator. Do NOT feed 5 mm here.
* HU are clipped to [--clip_lo, --clip_hi] (default -1000..200) to match training.
* Shapes are kept native (D,H,W); patches are cropped during training.
"""
import os, sys, glob, argparse
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sys.exit("SimpleITK is required:  pip install SimpleITK")


def _to_hu_array(img, clip, rescale=None):
    """SimpleITK image -> float32 HU numpy (D,H,W), clipped.

    `rescale=(slope, intercept)` un-normalises datasets that ship intensities
    already mapped to [0,1] instead of HU (RPLHR-CT: HU = v*3072 - 1024).
    """
    a = sitk.GetArrayFromImage(img).astype(np.float32)   # (D,H,W)
    if rescale:
        a = a * rescale[0] + rescale[1]
    lo, hi = clip
    return np.clip(a, lo, hi)


def _is_ct_like(a):
    """Cheap sanity check: CT lung volumes span air (~-1000) to soft tissue."""
    return a.ndim == 3 and min(a.shape) >= 8 and a.min() <= -500


def lung_crop(a, margin=(4, 12, 12), min_frac=0.005):
    """Crop to the lung bounding box: `a[lung_bbox(a)]`. See lung_bbox."""
    return a[lung_bbox(a, margin, min_frac)]


def lung_bbox(a, margin=(4, 12, 12), min_frac=0.005):
    """Bounding box (tuple of slices) of the lungs (optional, --lung_crop).

    Training samples random patches from whatever is stored here; on a full
    512x512 chest volume most patches land outside the body, where LAA-950 is a
    trivial 100% (clipped air) and carries no densitometry signal. Cropping to
    the lungs keeps the patches — and therefore the LAA/Perc15 supervision —
    inside parenchyma. Falls back to the whole volume if no lung is found.
    """
    from scipy import ndimage
    whole = tuple(slice(0, s) for s in a.shape)
    air = a < -400.0
    lbl, n = ndimage.label(air)
    if n == 0:
        return whole
    # air touching an in-plane border is outside-body air / table gap, not lung
    # (z borders are excluded: the trachea and lung apices may touch them)
    outside = np.unique(np.concatenate([lbl[:, 0, :].ravel(), lbl[:, -1, :].ravel(),
                                        lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()]))
    sizes = ndimage.sum_labels(air, lbl, index=np.arange(1, n + 1))
    keep = [i + 1 for i in range(n)
            if (i + 1) not in outside and sizes[i] > min_frac * a.size]
    if not keep:
        return whole
    mask = np.isin(lbl, keep)
    idx = [np.where(mask.any(axis=tuple(j for j in range(3) if j != ax)))[0]
           for ax in range(3)]
    return tuple(slice(max(0, i[0] - m), min(s, i[-1] + 1 + m))
                 for i, m, s in zip(idx, margin, a.shape))


def read_file(path):
    return sitk.ReadImage(path)


def read_dicom_series(folder):
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(folder)
    if not ids:
        return None
    # pick the series with the most slices (usually the full axial thin series)
    best, best_n = None, -1
    for sid in ids:
        files = reader.GetGDCMSeriesFileNames(folder, sid)
        if len(files) > best_n:
            best, best_n = files, len(files)
    reader.SetFileNames(best)
    return reader.Execute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input dir")
    ap.add_argument("--out", required=True, help="output dir for .npy")
    ap.add_argument("--dicom", action="store_true",
                    help="treat each sub-folder of --in as one DICOM series")
    ap.add_argument("--clip_lo", type=float, default=-1000.0)
    ap.add_argument("--clip_hi", type=float, default=200.0)
    ap.add_argument("--min_slices", type=int, default=40,
                    help="skip volumes thinner than this many slices")
    ap.add_argument("--rescale_hu", type=float, nargs=2, metavar=("SLOPE", "INTERCEPT"),
                    default=None,
                    help="HU = value*SLOPE + INTERCEPT, for sets shipped in "
                         "normalised units (RPLHR-CT: --rescale_hu 3072 -1024)")
    ap.add_argument("--lung_crop", action="store_true",
                    help="crop each volume to its lung bounding box (recommended: "
                         "keeps training patches inside parenchyma)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many written volumes (0 = all)")
    args = ap.parse_args()
    clip = (args.clip_lo, args.clip_hi)
    os.makedirs(args.out, exist_ok=True)

    jobs = []  # (name, kind, path)
    if args.dicom:
        for d in sorted(os.listdir(args.inp)):
            p = os.path.join(args.inp, d)
            if os.path.isdir(p):
                jobs.append((d, "dicom", p))
    else:
        for p in sorted(glob.glob(os.path.join(args.inp, "**", "*.nii*"), recursive=True) +
                        glob.glob(os.path.join(args.inp, "**", "*.mhd"), recursive=True)):
            name = os.path.splitext(os.path.basename(p))[0].replace(".nii", "")
            jobs.append((name, "file", p))

    if not jobs:
        sys.exit(f"no inputs found under {args.inp} "
                 f"({'DICOM sub-folders' if args.dicom else '.nii/.nii.gz/.mhd'})")

    ok = skip = 0
    for name, kind, path in jobs:
        try:
            img = read_dicom_series(path) if kind == "dicom" else read_file(path)
            if img is None:
                print(f"[skip] {name}: no readable series"); skip += 1; continue
            a = _to_hu_array(img, clip, args.rescale_hu)
            if a.shape[0] < args.min_slices or not _is_ct_like(a):
                print(f"[skip] {name}: shape {a.shape} not CT-like/too thin"); skip += 1; continue
            raw_shape = a.shape
            if args.lung_crop:
                a = lung_crop(a)
                if a.shape[0] < args.min_slices:
                    print(f"[skip] {name}: lung crop {a.shape} too thin"); skip += 1; continue
            outp = os.path.join(args.out, f"{name}.npy")
            np.save(outp, np.ascontiguousarray(a))
            ok += 1
            crop_note = f" (lung crop from {raw_shape})" if args.lung_crop else ""
            print(f"[ok]   {name}: {a.shape}{crop_note}  "
                  f"HU[{a.min():.0f},{a.max():.0f}] -> {outp}")
            if args.limit and ok >= args.limit:
                print(f"[stop] --limit {args.limit} reached"); break
        except Exception as e:
            print(f"[skip] {name}: {e}"); skip += 1

    print(f"\ndone: {ok} written, {skip} skipped -> {args.out}")
    if ok:
        print("next: python -m scte_r.train --data simulated --root "
              f"{args.out} --epochs 50 --ckpt public.pt")


if __name__ == "__main__":
    main()
