"""SCTE-R: assembles M1-M5 into one model.

Input:  observed 5 mm thick volume y (B,1,Dt,H,W), optional kernel_id.
Output: dict with x_hat (1 mm recon), cond, dc_residual, reliability.
"""
import torch
import torch.nn as nn
from .forward_operator import SSPForwardOperator
from .modules import OperatorNoiseID, ReconBackbone, DataConsistency, Reliability


class SCTE_R(nn.Module):
    def __init__(self, downsample=5, slice_fwhm_mm=5.0, base_ch=32, n_blocks=6,
                 cond_dim=32, dc_steps=1, n_kernels=16):
        super().__init__()
        self.downsample = downsample
        self.op = SSPForwardOperator(slice_fwhm_mm=slice_fwhm_mm, downsample=downsample)
        self.m1 = OperatorNoiseID(n_kernels=n_kernels, cond_dim=cond_dim)
        self.m2 = ReconBackbone(base_ch=base_ch, n_blocks=n_blocks, cond_dim=cond_dim)
        self.m3 = DataConsistency(self.op, steps=dc_steps)
        self.m5 = Reliability()

    def forward(self, y_thick, kernel_id=None):
        B, C, Dt, H, W = y_thick.shape
        D = Dt * self.downsample
        cond, w0, sigma = self.m1(y_thick, kernel_id)
        y_up = self.op.upsample_to_grid(y_thick, D)          # M2 input on 1mm grid
        x_hat = self.m2(y_up, cond)                          # residual recon
        x_hat = self.m3(x_hat, y_thick, sigma_vox=w0)        # data-consistency projection
        dc_res = self.m3.residual(x_hat, y_thick, sigma_vox=w0)
        rel = self.m5(dc_res)
        return dict(x_hat=x_hat, cond=cond, w0=w0, noise_sigma=sigma,
                    dc_residual=dc_res, reliability=rel)


def build_model(iters=1, arc=False, flow=False, **kw):
    """Ablation switch:
       iters==1, arc=False -> single-pass SCTE_R (the packaged variant)
       arc=True            -> ARC-CT: same network, acquisition identified per scan,
                              reconstruction accompanied by a certificate
       iters>1             -> the iterative/agentic variant (ARC-CT, re-identifying)
    """
    if iters and iters > 1:
        from .agentic import SCTER_Iterative
        return SCTER_Iterative(iters=iters, **kw)
    if flow:
        from .flow import FlowDecoder
        return FlowDecoder(**kw)
    if arc:
        from .agentic import ARC_CT
        return ARC_CT(**kw)
    kw.pop("residual_scale", None); kw.pop("steps", None); kw.pop("learned_op", None)
    kw.pop("guidance", None)
    kw.pop("op_source", None); kw.pop("calibration", None)
    kw.pop("e_mask", None); kw.pop("e_z", None); kw.pop("identify", None)
    return SCTE_R(**kw)
