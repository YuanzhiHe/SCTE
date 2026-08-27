"""
SSP forward operator: the shared physical degradation model 1mm -> 5mm.

The SAME operator is used two ways:
  (1) Simulation (public pretraining): take a thin (1mm) volume, apply A() to
      synthesize a 5mm-consistent thick volume with controllable slice width.
  (2) Data consistency (M3): re-blur the reconstruction and compare to the
      observed 5mm (HU/mass conservation).

Volumes are (B, 1, D, H, W); z is the through-plane axis (dim=2).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, radius: int = None):
    if radius is None:
        radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def fwhm_to_sigma(fwhm_mm: float, voxel_mm: float = 1.0) -> float:
    return (fwhm_mm / 2.3548) / voxel_mm


class SSPForwardOperator(nn.Module):
    """A(x): Gaussian slice-sensitivity blur along z, then downsample by r.

    Args:
        slice_fwhm_mm: effective slice width w0 of the thick series (e.g. 5 mm).
        voxel_z_mm:    thin-slice spacing (e.g. 1 mm).
        downsample:    integer z-downsample factor (thick_spacing / thin_spacing).
    """

    def __init__(self, slice_fwhm_mm: float = 5.0, voxel_z_mm: float = 1.0,
                 downsample: int = 5):
        super().__init__()
        self.slice_fwhm_mm = slice_fwhm_mm
        self.voxel_z_mm = voxel_z_mm
        self.downsample = downsample

    def blur_z(self, x: torch.Tensor, sigma_vox: torch.Tensor) -> torch.Tensor:
        """Depthwise Gaussian blur along z. sigma_vox may be a per-sample tensor."""
        B = x.shape[0]
        out = torch.empty_like(x)
        for b in range(B):
            s = float(sigma_vox[b]) if torch.is_tensor(sigma_vox) else float(sigma_vox)
            k = gaussian_kernel1d(max(s, 1e-3)).to(x.device, x.dtype)
            r = (k.numel() - 1) // 2
            ker = k.view(1, 1, -1, 1, 1)
            xb = F.pad(x[b:b + 1], (0, 0, 0, 0, r, r), mode="replicate")
            out[b:b + 1] = F.conv3d(xb, ker)
        return out

    def forward(self, x: torch.Tensor, sigma_vox: torch.Tensor = None) -> torch.Tensor:
        if sigma_vox is None:
            sigma_vox = fwhm_to_sigma(self.slice_fwhm_mm, self.voxel_z_mm)
        xb = self.blur_z(x, sigma_vox)
        # z-downsample by strided average (approx. thick-slice sampling)
        r = self.downsample
        if r > 1:
            xb = F.avg_pool3d(xb, kernel_size=(r, 1, 1), stride=(r, 1, 1))
        return xb

    def upsample_to_grid(self, y_thick: torch.Tensor, out_depth: int) -> torch.Tensor:
        """Trilinear up-sample a thick volume back to the 1 mm z-grid (network input)."""
        B, C, D, H, W = y_thick.shape
        return F.interpolate(y_thick, size=(out_depth, H, W),
                             mode="trilinear", align_corners=False)


def simulate_thick_from_thin(x_thin_hu: torch.Tensor, slice_fwhm_mm: float,
                             downsample: int, thin_noise_sigma_hu: float = 0.0,
                             thick_noise_sigma_hu: float = 0.0):
    """Public-pretraining helper: build a (thick, thin) supervised pair.

    Returns (y_thick, x_thin) both in HU. thick has less noise (thicker slab).
    Add noise in HU domain; thin noise > thick noise (physics).
    """
    op = SSPForwardOperator(slice_fwhm_mm=slice_fwhm_mm, downsample=downsample)
    x = x_thin_hu
    if thin_noise_sigma_hu > 0:
        x = x + torch.randn_like(x) * thin_noise_sigma_hu
    y = op(x)
    if thick_noise_sigma_hu > 0:
        y = y + torch.randn_like(y) * thick_noise_sigma_hu
    return y, x
