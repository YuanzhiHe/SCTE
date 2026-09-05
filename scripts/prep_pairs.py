"""Build REAL thin/thick training pairs from a paired public set (RPLHR-CT).

Everything so far trained on thick volumes synthesised by the SSP operator. This
script prepares the other kind of supervision: the *scanner's own* thick series,
aligned to its thin counterpart, so `public.pt` can be fine-tuned on real
degradation. It is also the dry run for the private-data pipeline — the same
three problems have to be solved there (z alignment, exact downsample ratio,
common lung crop), and the per-case audit it writes mirrors `pair_audit.csv`.

Output (under --out), for every case that passes:
    <case>_thin.npy    float32 HU (D, H, W)
    <case>_thick.npy   float32 HU (D/r, H, W)          exactly D = r * Dthick
    pair_audit.csv     case, offset, shapes, match error, lung fraction

Usage
-----
  python scripts/prep_pairs.py --thin DATA/rplhr/train/1mm --thick DATA/rplhr/train/5mm \
         --out DATA/public_pairs --rescale_hu 3072 -1024 --downsample 5
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import SimpleITK as sitk
from prep_public import lung_bbox
from scte_r.forward_operator import SSPForwardOperator

HU_CLIP = (-1000.0, 200.0)


def read_hu(path, rescale):
    """NIfTI/mhd file OR a DICOM series directory -> float32 HU (D,H,W), clipped."""
    if os.path.isdir(path):
        img = read_dicom_series(path)
        if img is None:
            raise IOError(f"no readable DICOM series in {path}")
    else:
        img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    if rescale:
        a = a * rescale[0] + rescale[1]
    return np.clip(a, *HU_CLIP)


def read_dicom_series(folder):
    """Largest CT series in `folder` (SimpleITK applies rescale slope/intercept)."""
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(folder)
    if not ids:
        return None
    best, n = None, -1
    for sid in ids:
        files = r.GetGDCMSeriesFileNames(folder, sid)
        if len(files) > n:
            best, n = files, len(files)
    r.SetFileNames(best)
    return r.Execute()


# DICOM tags describing the SCANNER AND THE ACQUISITION - never the patient.
# This is a strict allow-list, not an exclusion list: adding a tag is a deliberate
# act, so no patient attribute can arrive by default. Free-text fields
# (SeriesDescription, ProtocolName, StudyDescription) are deliberately NOT here -
# badly configured sites put names and IDs in them.
PROTOCOL_TAGS = {
    "0008|0070": "manufacturer",
    "0008|1090": "model",
    "0018|1210": "kernel",             # ConvolutionKernel
    "0018|0060": "kvp",
    "0018|1151": "tube_current_mA",
    "0018|1152": "exposure_mAs",
    "0018|0050": "slice_thickness_mm",
    "0018|0088": "spacing_between_slices_mm",
    "0018|1100": "recon_diameter_mm",
    "0018|1160": "filter_type",
    "0018|1020": "software_version",
    "0018|9305": "revolution_time_s",
    "0018|9311": "spiral_pitch",
    # Acquisition identity, not patient identity: these tell whether the two series are
    # two reconstructions of ONE acquisition or two separate scans. A hospital archive
    # does not guarantee the former, and nothing else in the header reveals it.
    "0020|0012": "acquisition_number",
    "0008|0022": "acquisition_date",
    "0008|0032": "acquisition_time",
    "0020|000e": "series_uid",
}


def protocol_fingerprint(path):
    """Scanner/acquisition metadata for one series, from the allow-list above.

    Recorded because the certificate's constants are PROTOCOL properties, not global
    ones: across the protocols measured so far the displacement threshold tau_delta
    spans 0.26 - 4.33 HU, a 17x range. Fitting them needs paired 1 mm data, which the
    deployment target (a hospital with only thick-slice CT) does not have - so the
    only route to certifying a scan there is to match its protocol against cohorts
    where the constants WERE measured. That match is impossible to reconstruct after
    the fact, hence capturing it now.
    """
    out = {}
    if not os.path.isdir(path):
        return out
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(path)
    if not ids:
        return out
    files, n = None, -1
    for sid in ids:
        f = r.GetGDCMSeriesFileNames(path, sid)
        if len(f) > n:
            files, n = f, len(f)
    try:
        rd = sitk.ImageFileReader()
        rd.SetFileName(files[0]); rd.LoadPrivateTagsOff(); rd.ReadImageInformation()
        for tag, name in PROTOCOL_TAGS.items():
            if rd.HasMetaDataKey(tag):
                v = rd.GetMetaData(tag).strip()
                if v:
                    out[name] = v
    except Exception as e:
        out["fingerprint_error"] = str(e)[:80]
    return out


def true_z_spacing(path):
    """Median gap between consecutive slice centres, from ImagePositionPatient.

    SliceThickness is not it. A "1 mm" series is often reconstructed at 0.7-0.8 mm
    increments and a "5 mm" one at 6-7 mm, so the index ratio between the two series is
    whatever those two numbers give - measured on one live cohort: median 6.404, range
    5.000-6.772, and not an integer. Hard-wiring downsample=5 then mis-pairs every slice
    by a growing amount while the content still matches at 0.997, which looks like a data
    problem and is not one.
    """
    if not os.path.isdir(path):
        img = sitk.ReadImage(path)
        return float(img.GetSpacing()[2])
    r = sitk.ImageSeriesReader()
    ids = r.GetGDCMSeriesIDs(path)
    if not ids:
        return float("nan")
    best, n = None, -1
    for sid in ids:
        f = r.GetGDCMSeriesFileNames(path, sid)
        if len(f) > n:
            best, n = f, len(f)
    zs = []
    for fn in best[:60]:
        rd = sitk.ImageFileReader(); rd.SetFileName(fn); rd.LoadPrivateTagsOff()
        rd.ReadImageInformation()
        if rd.HasMetaDataKey("0020|0032"):
            zs.append(float(rd.GetMetaData("0020|0032").split("\\")[2]))
    if len(zs) < 3:
        return float("nan")
    return float(np.median(np.diff(np.sort(np.array(zs)))))


def resample_z(vol, src_mm, dst_mm):
    """Linear resampling along z only, so the pair lands on an exact integer ratio."""
    if abs(src_mm - dst_mm) < 1e-4:
        return vol
    n_out = max(int(round(vol.shape[0] * src_mm / dst_mm)), 4)
    src = np.arange(vol.shape[0], dtype=np.float64) * src_mm
    dst = np.arange(n_out, dtype=np.float64) * dst_mm
    idx = np.clip(np.searchsorted(src, dst) - 1, 0, vol.shape[0] - 2)
    w = ((dst - src[idx]) / (src[idx + 1] - src[idx]))[:, None, None].astype(np.float32)
    return (vol[idx] * (1 - w) + vol[idx + 1] * w).astype(np.float32)


def series_geometry(path):
    """(n_slices, z_spacing_mm) for a file or a DICOM series directory."""
    img = read_dicom_series(path) if os.path.isdir(path) else sitk.ReadImage(path)
    if img is None:
        return 0, 0.0
    return img.GetSize()[2], float(img.GetSpacing()[2])


def discover_cases(root, thin_hint=None, thick_hint=None):
    """One sub-directory per case, each holding TWO series (dir or file).

    The thin/thick roles are decided by z-spacing when the DICOM headers carry it,
    and by slice count otherwise (more slices = thinner) — the private cohorts are
    1 mm / 5 mm, so the ratio is unambiguous either way. Name hints override.
    """
    cases = []
    for d in sorted(os.listdir(root)):
        cdir = os.path.join(root, d)
        if not os.path.isdir(cdir):
            continue
        members = [os.path.join(cdir, m) for m in sorted(os.listdir(cdir))]
        members = [m for m in members if os.path.isdir(m) or
                   m.lower().endswith((".nii", ".nii.gz", ".mhd"))]
        if len(members) < 2:
            print(f"[skip] {d}: found {len(members)} series, need 2"); continue
        if thin_hint and thick_hint:
            thin = [m for m in members if thin_hint.lower() in os.path.basename(m).lower()]
            thick = [m for m in members if thick_hint.lower() in os.path.basename(m).lower()]
            if len(thin) == 1 and len(thick) == 1:
                cases.append((d, thin[0], thick[0])); continue
            print(f"[warn] {d}: name hints did not resolve, falling back to geometry")
        geo = [(m,) + series_geometry(m) for m in members]
        geo = [g for g in geo if g[1] > 0]
        if len(geo) < 2:
            print(f"[skip] {d}: unreadable series"); continue
        # prefer z-spacing when both report it, else slice count
        if all(g[2] > 0 for g in geo[:2]):
            geo.sort(key=lambda g: g[2])                     # thinnest spacing first
        else:
            geo.sort(key=lambda g: -g[1])                    # most slices first
        cases.append((d, geo[0][0], geo[-1][0]))
        print(f"[pair] {d}: thin={os.path.basename(geo[0][0])} "
              f"({geo[0][1]} sl, {geo[0][2]:.2f} mm) | "
              f"thick={os.path.basename(geo[-1][0])} ({geo[-1][1]} sl, {geo[-1][2]:.2f} mm)")
    return cases


def z_offset(thin, thick, op, search=4, r=5):
    """Alignment minimising ||A(thin) - thick||, searched at THIN-slice resolution.

    The earlier version searched whole thick slices only, so any residual shift
    smaller than one slab was invisible and stayed in the data forever. On the
    public cohort that left a systematic 2-thin-slice offset: the operator's slabs
    were centred 2 slices away from the thick slice they were supposed to explain,
    which cost 29% of the forward-model residual and ~3 dB on every interpolation
    baseline. Sub-slab search is not optional - the two series are reconstructed
    independently and there is no reason for them to land on the same grid.

    Returns (slab_offset, sub_offset, err); the caller starts thin at
    r*slab_offset + sub_offset.
    """
    best, best_err = (0, 0), float("inf")
    for sub in range(-r + 1, r):
        t0 = max(sub, 0)
        thin_s = thin[t0:] if sub >= 0 else np.concatenate(
            [np.repeat(thin[:1], -sub, axis=0), thin], axis=0)
        n_t = thin_s.shape[0] // r
        if n_t < 4:
            continue
        sim = op(torch.from_numpy(thin_s[:n_t * r][None, None]) / 1000.0)[0, 0].numpy()
        n = min(sim.shape[0], thick.shape[0])
        for off in range(-search, search + 1):
            a0, b0 = max(0, off), max(0, -off)
            m = min(n - a0, n - b0)
            if m < 4:
                continue
            err = float(np.mean((sim[a0:a0 + m] - thick[b0:b0 + m] / 1000.0) ** 2))
            if err < best_err:
                best, best_err = (off, sub), err
    return best[0], best[1], best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thin", default=None); ap.add_argument("--thick", default=None)
    ap.add_argument("--cases", default=None,
                    help="PRIVATE-DATA MODE: one sub-directory per case, each holding "
                         "the thin and the thick series (DICOM dirs or NIfTI files). "
                         "Roles are resolved by z-spacing, else by slice count.")
    ap.add_argument("--thin_hint", default=None, help="e.g. '1mm' — overrides geometry")
    ap.add_argument("--thick_hint", default=None, help="e.g. '5mm'")
    ap.add_argument("--anonymise", action="store_true",
                    help="write cases as sha1(case-id)[:12] and keep the mapping in a "
                         "SEPARATE file that never leaves the secure machine")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cases (dry run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rescale_hu", type=float, nargs=2, default=None)
    ap.add_argument("--downsample", type=int, default=5)
    ap.add_argument("--slice_fwhm", type=float, default=5.0)
    ap.add_argument("--min_slices", type=int, default=40)
    ap.add_argument("--max_offset", type=int, default=4)
    ap.add_argument("--fix_ratio", action="store_true", default=True,
                    help="resample the thin series so thick/thin spacing is exactly "
                         "--downsample. Off by --no_fix_ratio.")
    ap.add_argument("--no_fix_ratio", dest="fix_ratio", action="store_false")
    ap.add_argument("--protocol_meta", action="store_true", default=True,
                    help="record scanner/acquisition metadata per case (allow-listed "
                         "DICOM tags only, never patient attributes)")
    ap.add_argument("--no_protocol_meta", dest="protocol_meta", action="store_false")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    r = args.downsample
    op = SSPForwardOperator(slice_fwhm_mm=args.slice_fwhm, downsample=r)

    if args.cases:
        jobs = discover_cases(args.cases, args.thin_hint, args.thick_hint)
    else:
        names = sorted(f for f in os.listdir(args.thin) if f.endswith((".nii", ".nii.gz")))
        jobs = [(nm.split(".")[0], os.path.join(args.thin, nm),
                 os.path.join(args.thick, nm)) for nm in names
                if os.path.exists(os.path.join(args.thick, nm))]
    if args.limit:
        jobs = jobs[:args.limit]
    rows, ok, skip, idmap = [], 0, 0, []
    for case_raw, thin_p, thick_p in jobs:
        case = case_raw
        if args.anonymise:
            import hashlib
            case = hashlib.sha1(case_raw.encode()).hexdigest()[:12]
            idmap.append((case, case_raw))
        try:
            thin = read_hu(thin_p, args.rescale_hu)
            thick = read_hu(thick_p, args.rescale_hu)
            sz_t, sz_k = true_z_spacing(thin_p), true_z_spacing(thick_p)
            ratio = sz_k / sz_t if (sz_t and sz_t == sz_t and sz_k == sz_k) else float("nan")
            if args.fix_ratio and ratio == ratio and abs(ratio - r) > 0.02:
                # Bring the thin series onto a grid where thick/thin is exactly r, so
                # every downstream assumption (operator, certificate, public weights)
                # keeps holding without change.
                thin = resample_z(thin, sz_t, sz_k / r)
                sz_t = sz_k / r
            bb = lung_bbox(thin)
            ys, xs = bb[1], bb[2]
            z_lo = bb[0].start - bb[0].start % r          # snap so slabs line up
            off, sub, err = z_offset(thin[:, ys, xs], thick[:, ys, xs], op, args.max_offset, r)
            # z_offset's contract, read straight off its search: thin slab k starts at
            #   z_lo = sub + max(off,0)*r + k*r      and pairs with thick[max(-off,0)+k].
            # Deriving the indices from that identity instead of patching a snapped
            # bbox start is what makes a NEGATIVE sub safe: the old form could land on
            # z_lo < 0, and numpy reads a negative slice start from the END of the array,
            # so the crop came back EMPTY and the case was silently skipped.
            z_lo = sub + max(off, 0) * r
            t0 = max(-off, 0)
            while z_lo < 0 or z_lo < bb[0].start - r:     # advance whole slabs into range
                z_lo += r; t0 += 1
            if t0 >= thick.shape[0] or z_lo + r > thin.shape[0]:
                print(f"[skip] {case}: no overlapping slab after alignment"); skip += 1; continue
            n_t = min((thin.shape[0] - z_lo) // r, thick.shape[0] - t0,
                      max(bb[0].stop - z_lo, 0) // r)
            if n_t * r < args.min_slices:
                print(f"[skip] {case}: overlap {n_t * r} slices too short"); skip += 1; continue
            x = np.ascontiguousarray(thin[z_lo:z_lo + n_t * r, ys, xs])
            y = np.ascontiguousarray(thick[t0:t0 + n_t, ys, xs])
            np.save(os.path.join(args.out, f"{case}_thin.npy"), x)
            np.save(os.path.join(args.out, f"{case}_thick.npy"), y)
            lung = float(((x > -990) & (x < -500)).mean())
            rec = dict(case=case, offset=off, sub_offset=sub, match_mse=round(err, 6),
                       thin_shape=str(x.shape), thick_shape=str(y.shape),
                       lung_frac=round(lung, 4),
                       z_thin_mm=round(sz_t, 4) if sz_t == sz_t else "",
                       z_thick_mm=round(sz_k, 4) if sz_k == sz_k else "",
                       z_ratio_raw=round(ratio, 4) if ratio == ratio else "")
            if args.protocol_meta:
                for role, pth in (("thin", thin_p), ("thick", thick_p)):
                    for k, v in protocol_fingerprint(pth).items():
                        rec[f"{role}_{k}"] = v
            rows.append(rec)
            ok += 1
            print(f"[ok]   {case}: thin {x.shape} thick {y.shape} off={off:+d} sub={sub:+d} "
                  f"z {sz_t:.2f}/{sz_k:.2f}mm ratio {ratio:.2f} "
                  f"mse={err:.5f} lung={lung:.2f} HU[{x.min():.0f},{x.max():.0f}]")
            if ok == 3 and all(r_["lung_frac"] < 0.02 for r_ in rows[:3]):
                print("\n!! the first 3 cases have essentially NO lung voxels.\n"
                      "!! Most likely the series are NOT in Hounsfield units — check the\n"
                      "!! printed HU range above: air should sit near -1000 and soft tissue\n"
                      "!! near 0. If the values look like [0,1], re-run with e.g.\n"
                      "!!     --rescale_hu 3072 -1024\n"
                      "!! (that is HU = value*3072 - 1024, the usual 12-bit CT packing).\n")
        except Exception as e:
            print(f"[skip] {case}: {e}"); skip += 1

    if idmap:
        with open(os.path.join(args.out, "ID_MAP_DO_NOT_EXPORT.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["anon_id", "original_id"]); w.writerows(idmap)
        print("wrote ID_MAP_DO_NOT_EXPORT.csv (keep it on the secure machine)")
    if rows:
        with open(os.path.join(args.out, "pair_audit.csv"), "w", newline="") as f:
            keys = list(rows[0].keys())
            for r_ in rows:                          # cases can differ in which tags exist
                keys += [k for k in r_ if k not in keys]
            w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader()
            w.writerows(rows)
    offs = [r_["offset"] for r_ in rows]
    print(f"\ndone: {ok} pairs, {skip} skipped -> {args.out}")
    if offs:
        print("z-offsets:", dict(zip(*np.unique(offs, return_counts=True))))


if __name__ == "__main__":
    main()
