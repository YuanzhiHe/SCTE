"""Emphysema densitometry + image-quality metrics.

Volumes are carried as kHU = HU / 1000 (see datasets.HU_SCALE), so the -950 HU
emphysema threshold is -0.95 and the data range is ~2.0. Multiply Perc15 by 1000
to report in HU.
"""
import torch
import torch.nn.functional as F

LAA_THRESH = -0.95          # -950 HU in kHU units
DATA_RANGE = 2.0            # ~ (200 - (-1000)) / 1000


def _lung_mask(hu, low=-1.0, high=-0.5):
    """Crude lung parenchyma mask by HU window. Replace with a segmentation mask
    (e.g. TotalSegmentator / lungmask) on real data."""
    return (hu >= low) & (hu <= high)


def laa950(hu, mask=None, thresh=LAA_THRESH):
    """% of lung voxels below thresh (emphysema index). Returns percentage."""
    if mask is None:
        mask = _lung_mask(hu)
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return 100.0 * ((hu < thresh).float() * m).sum() / denom


def perc15(hu, mask=None):
    """15th percentile of lung HU (Perc15)."""
    return percN(hu, 0.15, mask)


def percN(hu, q, mask=None):
    """q-th percentile of lung HU — Perc15/Perc10/Perc5 are q = 0.15/0.10/0.05."""
    if mask is None:
        mask = _lung_mask(hu)
    vals = hu[mask]
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=hu.device)
    return torch.quantile(vals.float(), q)


def laaN(hu, thresh, mask=None):
    """% of lung voxels below `thresh` (kHU). LAA-950/LAA-910 are -0.95 / -0.91."""
    if mask is None:
        mask = _lung_mask(hu)
    m = mask.float()
    return 100.0 * ((hu < thresh).float() * m).sum() / m.sum().clamp_min(1.0)


def laa910(hu, mask=None):
    return laaN(hu, -0.91, mask)


def laa_soft(hu, mask=None, thresh=LAA_THRESH, tau=0.005):
    """Differentiable surrogate of LAA-950 (sigmoid) for use inside losses."""
    if mask is None:
        mask = _lung_mask(hu)
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    frac = (torch.sigmoid((thresh - hu) / tau) * m).sum() / denom
    return 100.0 * frac


def psnr(pred_hu, target_hu, data_range=DATA_RANGE):
    mse = F.mse_loss(pred_hu, target_hu)
    return 10.0 * torch.log10((data_range ** 2) / mse.clamp_min(1e-8))


def ssim(pred, target, data_range=DATA_RANGE, win=7):
    """3D SSIM (global-window approximation). For paper metrics use a validated
    library; this is a lightweight in-repo version for smoke tests / monitoring."""
    mu_x = pred.mean(); mu_y = target.mean()
    vx = pred.var(); vy = target.var()
    vxy = ((pred - mu_x) * (target - mu_y)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * vxy + c2)) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2))


def ccc(a, b):
    """Lin's concordance correlation coefficient over a batch of scalars."""
    a = a.float(); b = b.float()
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - ma) * (b - mb)).mean()
    return (2 * cov) / (va + vb + (ma - mb) ** 2 + 1e-8)


def nps_1d_z(vol):
    """Noise power spectrum along z (through-plane), averaged over x/y.
    Use a homogeneous-ROI version on real data; here a global proxy for the loss."""
    # remove low-freq structure with a difference along z, then power spectrum
    dz = vol[..., 1:, :, :] - vol[..., :-1, :, :]
    fz = torch.fft.rfft(dz, dim=2)
    return (fz.abs() ** 2).mean(dim=(0, 1, 3, 4))
