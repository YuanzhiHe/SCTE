"""Datasets for the two-stage plan.

  SyntheticPairedDataset : random lung-like volumes -> smoke tests, no data needed.
  SimulatedThickDataset  : PUBLIC thin-slice CT (LIDC/LUNA16/RPLHR thin) with the
                           5 mm thick simulated on the fly -> pretraining, perfect GT.
  RealPairedDataset      : PRIVATE real 1mm/5mm scanner pairs (from pair_audit.csv)
                           -> validation / fine-tuning, the claim that matters.

All datasets return (y_thick, x_thin, kernel_id) with volumes in HU, (1,D,H,W).
"""
import os
import csv
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from .forward_operator import SSPForwardOperator, fwhm_to_sigma

HU_CLIP = (-1000.0, 200.0)
HU_SCALE = 1000.0        # volumes carried as kHU = HU / 1000 (so -950 HU = -0.95)


def _to_tensor(vol_hu):
    v = np.clip(vol_hu, *HU_CLIP).astype(np.float32) / HU_SCALE
    return torch.from_numpy(v)[None]                     # (1,D,H,W) in kHU


# --------------------------------------------------------------------------
class SyntheticPairedDataset(Dataset):
    """Lung-like HU volumes for pipeline smoke tests (no data required)."""

    def __init__(self, n=32, shape=(40, 48, 48), downsample=5, slice_fwhm=5.0, seed=0):
        self.n, self.shape = n, shape
        self.op = SSPForwardOperator(slice_fwhm_mm=slice_fwhm, downsample=downsample)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        D, H, W = self.shape
        g = np.random.default_rng(i + 1)
        # normal parenchyma ~ -860, low-attenuation tail ~ -930
        base = g.normal(-860, 30, (D, H, W))
        emph = g.normal(-940, 35, (D, H, W))
        mask = g.random((D, H, W)) < 0.25
        vol = np.where(mask, emph, base)
        x = _to_tensor(vol)
        y = self.op(x[None])[0]                          # simulate thick
        return y, x, torch.tensor(0, dtype=torch.long)


# --------------------------------------------------------------------------
class SimulatedThickDataset(Dataset):
    """Public thin-slice volumes (.npy in HU) -> simulate thick on the fly.

    Put preprocessed thin volumes as .npy (HU, shape (D,H,W)) under `root`.
    Use scripts/prepare_public.py (see README) to convert LIDC/LUNA16/RPLHR.
    """

    LUNG_STRIDE = 8          # coarse grid used to locate parenchyma

    def __init__(self, root, patch=(40, 64, 64), downsample=5, slice_fwhm_mm=5.0,
                 fwhm_range=None, thin_noise=15.0, thick_noise=7.0, seed=0,
                 random_crop=False, min_lung=0.25):
        self.files = self._list_files(root)
        self.patch = patch
        self.min_lung = min_lung
        self.downsample = downsample
        # the effective slice width is jittered around the nominal one so the
        # model does not overfit a single SSP (M1 has to identify it)
        self.fwhm_range = tuple(fwhm_range) if fwhm_range else \
            (slice_fwhm_mm, slice_fwhm_mm + 2.0)
        # noise sigmas are given in HU; volumes are carried in kHU (see HU_SCALE)
        self.thin_noise, self.thick_noise = thin_noise, thick_noise
        self.rng = np.random.default_rng(seed)
        # train: fresh crop/noise draw every epoch. eval: fixed per-index draw.
        self.random_crop = random_crop
        # patches must contain parenchyma, otherwise LAA-950/Perc15 carry no
        # signal (a patch of chest wall or of clipped air is 0% / 100% for both
        # the thick input and the reference). Cache one coarse lung map per case.
        self._lung = [self._lung_score_map(f) for f in self.files] if self.files else []

    @staticmethod
    def _list_files(root):
        return [os.path.join(root, f) for f in sorted(os.listdir(root))
                if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def _lung_score_map(self, path):
        """Fraction of parenchyma inside every candidate patch, on a coarse grid.

        Cached next to the volume; -1000 HU (clipped air: outside the body, and
        airway lumen) is excluded so only real lung tissue counts.
        """
        tag = "x".join(str(p) for p in self.patch)      # the map is patch-specific
        cache = os.path.join(os.path.dirname(path), "_lungcache",
                             os.path.basename(path)[:-4] + f".{tag}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        s = self.LUNG_STRIDE
        v = np.load(path, mmap_mode="r")[::s, ::s, ::s]
        lung = ((v > -990.0) & (v < -500.0)).astype(np.float32)
        box = tuple(max(1, p // s) for p in self.patch)
        score = ndimage.uniform_filter(lung, size=box, mode="constant")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        try:
            np.save(cache, score)
        except OSError:
            pass
        return score

    def _crop_origin(self, shape, i, g):
        """Top-left corner of the patch: lung-bearing, random or deterministic."""
        s = self.LUNG_STRIDE
        score = self._lung[i]
        box = tuple(max(1, p // s) for p in self.patch)
        if self.random_crop:
            cand = np.argwhere(score >= self.min_lung)
            c = cand[g.integers(len(cand))] if len(cand) else \
                np.array(np.unravel_index(score.argmax(), score.shape))
        else:                                  # deterministic: most lung-filled
            c = np.array(np.unravel_index(score.argmax(), score.shape))
        start = [int((ci - bi // 2) * s) for ci, bi in zip(c, box)]
        if self.random_crop:                   # jitter inside the coarse cell
            start = [st + int(g.integers(0, s)) for st in start]
        return [int(np.clip(st, 0, max(0, dim - p)))
                for st, dim, p in zip(start, shape, self.patch)]

    def _crop(self, vol, i, g):
        z, y, x = self._crop_origin(vol.shape, i, g)
        pd, ph, pw = self.patch
        return vol[z:z + pd, y:y + ph, x:x + pw]

    def __getitem__(self, i):
        g = self.rng if self.random_crop else np.random.default_rng(i + 12345)
        # mmap: only the cropped patch is read from disk (volumes are ~100-300 MB)
        vol = self._crop(np.load(self.files[i], mmap_mode="r"), i, g)
        x = _to_tensor(np.ascontiguousarray(vol))
        fwhm = float(g.uniform(*self.fwhm_range))
        op = SSPForwardOperator(slice_fwhm_mm=fwhm, downsample=self.downsample)
        # noise in HU -> kHU. thin noise is blurred/averaged away by A(); the thick
        # series keeps its own (smaller) readout noise on top.
        gen = None if self.random_crop else torch.Generator().manual_seed(i + 12345)
        xn = x + torch.randn(x.shape, generator=gen) * (self.thin_noise / HU_SCALE)
        y = op(xn[None])[0]
        y = y + torch.randn(y.shape, generator=gen) * (self.thick_noise / HU_SCALE)
        return y, x, torch.tensor(0, dtype=torch.long)


# --------------------------------------------------------------------------
class PreparedPairDataset(SimulatedThickDataset):
    """REAL thin/thick pairs prepared by scripts/prep_pairs.py.

    Same patch sampling as SimulatedThickDataset, but the thick slab is the
    scanner's own reconstruction instead of A(x): no SSP model, no synthetic
    noise. `prep_pairs.py` has already z-aligned and trimmed the pair so that
    D_thin = downsample * D_thick, which is what the model assumes.
    """

    #: when True, also return the cached base reconstruction `<case>_base.npy`
    #: (a frozen strong backbone) so the generative stage can model the residual
    #: it leaves behind instead of the residual left by plain up-sampling.
    with_base = False

    @staticmethod
    def _list_files(root):
        return [os.path.join(root, f) for f in sorted(os.listdir(root))
                if f.endswith("_thin.npy")]

    def __getitem__(self, i):
        g = self.rng if self.random_crop else np.random.default_rng(i + 12345)
        thin = np.load(self.files[i], mmap_mode="r")
        thick = np.load(self.files[i].replace("_thin.npy", "_thick.npy"), mmap_mode="r")
        r = self.downsample
        pd, ph, pw = self.patch
        z, y, x = self._crop_origin(thin.shape, i, g)
        z -= z % r                                   # keep thin/thick slabs aligned
        zt = z // r
        x_thin = _to_tensor(np.ascontiguousarray(thin[z:z + pd, y:y + ph, x:x + pw]))
        y_thick = _to_tensor(np.ascontiguousarray(
            thick[zt:zt + pd // r, y:y + ph, x:x + pw]))
        if not self.with_base:
            return y_thick, x_thin, torch.tensor(0, dtype=torch.long)
        base = np.load(self.files[i].replace("_thin.npy", "_base.npy"), mmap_mode="r")
        b = _to_tensor(np.ascontiguousarray(
            base[z:z + pd, y:y + ph, x:x + pw]).astype(np.float32))
        return y_thick, x_thin, torch.tensor(0, dtype=torch.long), b


# --------------------------------------------------------------------------
def load_series(series_dir):
    """Load a DICOM series directory -> HU volume (D,H,W), sorted by z.
    Mirrors the verified audit loader (no *.dcm filter, per-file HU)."""
    import pydicom
    from pydicom.misc import is_dicom
    paths = []
    for f in os.listdir(series_dir):
        p = os.path.join(series_dir, f)
        if os.path.isfile(p) and f.upper() != "DICOMDIR" and is_dicom(p):
            paths.append(p)
    slices = [pydicom.dcmread(p) for p in paths]
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices])
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    inter = float(getattr(slices[0], "RescaleIntercept", -1024.0))
    return vol * slope + inter


class RealPairedDataset(Dataset):
    """Private real pairs from pair_audit.csv (columns thin_series/thick_series
    are folder names under each case dir). data_root points at the case dirs."""

    def __init__(self, audit_csv, data_root, patch=(40, 64, 64)):
        self.rows = [r for r in csv.DictReader(open(audit_csv, encoding="utf-8-sig"))
                     if r["has_pair"] == "YES"]
        self.data_root = data_root
        self.patch = patch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        case = os.path.join(self.data_root, r["dir"])
        x = _to_tensor(load_series(os.path.join(case, r["thin_series"])))
        y = _to_tensor(load_series(os.path.join(case, r["thick_series"])))
        # TODO: in-plane resample to common spacing; z-align; lung mask.
        return y, x, torch.tensor(0, dtype=torch.long)
