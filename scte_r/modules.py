"""SCTE-R network modules M1-M5.

M1  OperatorNoiseID     : estimate effective slice width w0, noise sigma, kernel emb.
M2  ReconBackbone       : FiLM-conditioned 3D residual net, predicts residual.
M3  DataConsistency     : SSP re-blur projection (HU/mass conservation).
M4  guardrail losses    : functional-trajectory + NPS calibration (in losses.py).
M5  Reliability         : per-scan reliability score from data-consistency residual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .forward_operator import SSPForwardOperator, fwhm_to_sigma


# ----------------------------- M1 -----------------------------
class OperatorNoiseID(nn.Module):
    """Identify (w0, noise_sigma, kernel-embedding) from the observed thick volume.

    Scaffold implementation:
      - noise sigma estimated from local z-differences in homogeneous regions.
      - w0 initialised from metadata (nominal), refined by a small CNN head.
      - kernel embedding is a learned vector keyed by a provided kernel id.
    TODO(research): replace the w0 head with the oversampled edge-response SSP fit
    at lung-diaphragm / apical interfaces (the paper's M1).
    """

    def __init__(self, n_kernels: int = 16, cond_dim: int = 32):
        super().__init__()
        self.kernel_emb = nn.Embedding(n_kernels, cond_dim // 2)
        self.enc = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.head = nn.Linear(8, cond_dim - cond_dim // 2)
        self.cond_dim = cond_dim

    @staticmethod
    def estimate_noise_sigma(thick):
        # robust std of z-differences (proxy for through-plane noise)
        dz = thick[..., 1:, :, :] - thick[..., :-1, :, :]
        return dz.flatten(1).std(dim=1) / (2 ** 0.5)

    def forward(self, thick, kernel_id=None, w0_nominal_fwhm=5.0):
        """w0_nominal_fwhm may be a scalar (nominal, the original behaviour) or a
        per-sample tensor of identified widths in mm (agentic.OperatorSolver)."""
        B = thick.shape[0]
        if kernel_id is None:
            kernel_id = torch.zeros(B, dtype=torch.long, device=thick.device)
        kemb = self.kernel_emb(kernel_id)
        feat = self.head(self.enc(thick))
        cond = torch.cat([kemb, feat], dim=1)
        sigma = self.estimate_noise_sigma(thick)                      # (B,)
        if torch.is_tensor(w0_nominal_fwhm):
            w0 = (w0_nominal_fwhm.to(thick.device).float() / 2.3548)  # fwhm -> sigma(vox)
        else:
            w0 = torch.full((B,), fwhm_to_sigma(w0_nominal_fwhm), device=thick.device)
        return cond, w0, sigma


# ----------------------------- FiLM ---------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim, n_ch):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, n_ch)
        self.to_shift = nn.Linear(cond_dim, n_ch)

    def forward(self, x, cond):
        s = self.to_scale(cond).view(x.shape[0], -1, 1, 1, 1)
        b = self.to_shift(cond).view(x.shape[0], -1, 1, 1, 1)
        return x * (1 + s) + b


class ResBlock3D(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.c1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.film = FiLM(cond_dim, ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, cond):
        h = self.act(self.c1(x))
        h = self.film(h, cond)
        h = self.c2(h)
        return self.act(x + h)


# ----------------------------- M2 -----------------------------
class ReconBackbone(nn.Module):
    """3D residual network conditioned on (w0/SSP/noise/kernel) via FiLM.
    Predicts a residual over the z-upsampled thick input."""

    def __init__(self, base_ch=32, n_blocks=6, cond_dim=32, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.inp = nn.Conv3d(in_ch, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock3D(base_ch, cond_dim) for _ in range(n_blocks)])
        self.out = nn.Conv3d(base_ch, 1, 3, padding=1)

    def forward(self, x_in, cond):
        # residual is taken over the FIRST input channel (the current estimate)
        base = x_in[:, :1]
        h = self.inp(x_in)
        for blk in self.blocks:
            h = blk(h, cond)
        return base + self.out(h)


# ----------------------------- M3 -----------------------------
class DataConsistency(nn.Module):
    """Gradient-step projection toward A(x)=y (HU/mass conservation).
    x <- x - lr * A^T(A x - y). A^T approximated by the same blur + z-upsample."""

    def __init__(self, operator: SSPForwardOperator, steps=1, lr=0.5):
        super().__init__()
        self.op = operator
        self.steps = steps
        self.lr = lr

    def forward(self, x, y_thick, sigma_vox=None):
        for _ in range(self.steps):
            Ax = self.op(x, sigma_vox)                       # (B,1,Dt,H,W)
            resid = Ax - y_thick
            grad = self.op.upsample_to_grid(resid, x.shape[2])  # A^T proxy
            x = x - self.lr * grad
        return x

    def residual(self, x, y_thick, sigma_vox=None):
        return (self.op(x, sigma_vox) - y_thick)


# ----------------------------- M5 -----------------------------
class Reliability(nn.Module):
    """Per-scan reliability score. Scaffold: inverse of data-consistency residual
    magnitude (higher = more trustworthy). TODO(research): leave-out-scale refit."""

    def forward(self, dc_residual):
        r = dc_residual.flatten(1).abs().mean(dim=1)
        return torch.exp(-r)               # in (0,1], 1 = reliable
