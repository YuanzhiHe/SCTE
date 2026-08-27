"""Flow-matching decoder for the tail-recovery half of the problem.

Why a generative decoder at all — this is an empirical conclusion, not a taste:
every deterministic objective we tried leaves the tail statistic FROZEN. Across
the displacement penalty (3 weights), a quantile-matching distribution loss
(3 weights) and 2 seeds, the certificate's tail term stayed at +5.90 sigma_v to
the last decimal, while only the histogram's POSITION moved. A conditional mean
estimator cannot do otherwise: given a smooth thick slab, the L2/L1-optimal thin
volume is smooth, and no reweighting of a per-voxel loss changes that.

So the decoder samples instead of averaging. Flow matching in RESIDUAL space:

    u = (x_thin - upsample(y_thick)) / s_r        s_r = residual_scale (kHU)
    u_t = (1 - t) * z + t * u,      z ~ N(0, I)
    the network predicts the velocity  v = u - z
    at inference, integrate from z with `steps` Euler steps, then project onto
    data consistency with the identified operator.

Residual space matters: the target has zero mean and O(50 HU) scale, so the noise
prior is on the same scale as the thing being synthesised — flowing all the way
from N(0,1) to a distribution centred at -0.88 kHU would spend every step
transporting the mean, which is exactly the quantity the certificate says is
already fine.
"""
import math

import torch
import torch.nn as nn

from .agentic import ARC_CT
from .modules import ReconBackbone


def time_embedding(t, dim=16):
    """Sinusoidal embedding of t in [0,1]; (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device))
    a = t[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class FlowDecoder(ARC_CT):
    """ARC-CT whose reconstruction solver samples rather than regresses.

    Everything else is unchanged — the same operator solver, the same certificate,
    the same arbiter — so the tail term is measured on exactly the same footing as
    the deterministic arms.
    """

    def __init__(self, residual_scale=0.05, steps=16, t_dim=16, guidance=0.0, **kw):
        super().__init__(**kw)
        self.s_r = residual_scale          # kHU; 0.05 = 50 HU, the lung texture scale
        self.steps = int(steps)
        # MEASUREMENT GUIDANCE. Projecting onto data consistency only at the END of
        # the trajectory leaves the sampler free to wander off the measurement for
        # the whole integration and then get yanked back once. Applying a small
        # gradient step on ||A_w(x) - y||^2 at EVERY Euler step keeps the trajectory
        # inside the measurement's neighbourhood, which is the standard fix for
        # diffusion/flow inverse problems. 0 = off (project only at the end).
        self.guidance = float(guidance)
        self.t_dim = t_dim
        cond_dim = self.net.m1.cond_dim
        # velocity net: consumes [u_t, y_up] and the (cond + time) vector
        self.vnet = ReconBackbone(base_ch=kw.get("base_ch", 32),
                                  n_blocks=kw.get("n_blocks", 6),
                                  cond_dim=cond_dim + t_dim, in_ch=2)

    def _cond(self, y_thick, kernel_id, w_hat, t):
        cond, w0, sigma = self.net.m1(y_thick, kernel_id, w0_nominal_fwhm=w_hat)
        return torch.cat([cond, time_embedding(t, self.t_dim)], dim=1), w0, sigma

    def velocity(self, u_t, y_up, cond):
        # ReconBackbone returns base + out(...); base is its FIRST input channel, so
        # feeding u_t first makes the net predict a residual over u_t. The velocity
        # we want is the correction ITSELF, hence the subtraction.
        return self.vnet(torch.cat([u_t, y_up], dim=1), cond) - u_t

    def flow_loss(self, y_thick, x_thin, kernel_id=None, base=None):
        """Conditional flow-matching objective (training path)."""
        B = y_thick.shape[0]
        D = y_thick.shape[2] * self.downsample
        w_hat = torch.full((B,), float(self.protocol_w), device=y_thick.device) \
            if self.identify else torch.full((B,), self.nominal_fwhm, device=y_thick.device)
        # `base` is an optional frozen strong reconstruction: the flow then models
        # only what that backbone leaves behind, which is where the density tail
        # lives. Falls back to plain up-sampling when no backbone is supplied.
        y_up = self.net.op.upsample_to_grid(y_thick, D) if base is None else base
        u = (x_thin - y_up) / self.s_r
        z = torch.randn_like(u)
        t = torch.rand(B, device=y_thick.device)
        tt = t.view(-1, 1, 1, 1, 1)
        u_t = (1 - tt) * z + tt * u
        cond, _, _ = self._cond(y_thick, kernel_id, w_hat, t)
        return (self.velocity(u_t, y_up, cond) - (u - z)).abs().mean()

    @torch.no_grad()
    def sample(self, y_thick, kernel_id=None, steps=None, generator=None, base=None):
        steps = steps or self.steps
        B = y_thick.shape[0]
        D = y_thick.shape[2] * self.downsample
        w_hat = torch.full((B,), float(self.protocol_w), device=y_thick.device) \
            if self.identify else torch.full((B,), self.nominal_fwhm, device=y_thick.device)
        y_up = self.net.op.upsample_to_grid(y_thick, D) if base is None else base
        u = torch.randn(y_up.shape, device=y_up.device, generator=generator)
        for k in range(steps):
            t = torch.full((B,), k / steps, device=y_up.device)
            cond, w0, _ = self._cond(y_thick, kernel_id, w_hat, t)
            u = u + self.velocity(u, y_up, cond) / steps
            if self.guidance > 0:
                x_k = y_up + self.s_r * u
                r = self.net.m3.op(x_k, w0) - y_thick
                grad = self.net.op.upsample_to_grid(r, x_k.shape[2])
                u = u - self.guidance * grad / self.s_r
        x_hat = y_up + self.s_r * u
        _, w0, _ = self._cond(y_thick, kernel_id, w_hat, torch.ones(B, device=y_up.device))
        return self.net.m3(x_hat, y_thick, sigma_vox=w0), w_hat      # data-consistent

    def forward(self, y_thick, kernel_id=None, base=None):
        x_hat, w_hat = self.sample(y_thick, kernel_id, base=base)
        dc_res = self.net.m3.residual(x_hat, y_thick,
                                      sigma_vox=w_hat.to(y_thick.device) / 2.3548)
        cert = self.densitometry(x_hat, y_thick, w_hat)
        B = y_thick.shape[0]
        return dict(x_hat=x_hat, w_hat=w_hat,
                    sigma_hat=torch.zeros(B, device=y_thick.device),
                    rho=torch.zeros(B, device=y_thick.device),
                    fit_at_edge=torch.zeros(B, dtype=torch.bool, device=y_thick.device),
                    dc_residual=dc_res, reliability=self.net.m5(dc_res),
                    cond=None, w0=w_hat / 2.3548,
                    noise_sigma=torch.zeros(B, device=y_thick.device), **cert)
