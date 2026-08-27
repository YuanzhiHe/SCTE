"""ARC-CT — Arbitrated Reconstruction with Certificates for CT.

The single-pass SCTE_R decides three things at once inside one network under one
scalar loss: what the acquisition did, what the anatomy was, and what the lung
histogram should look like. Because the map from a reconstruction to a threshold
statistic (LAA-950, Perc15) is many-to-one, that design cannot say WHY its
densitometric value came out right — a global attenuation displacement and a
genuine recovery of the low-attenuation tail are indistinguishable to it. We
measured the consequence: fine-tuning on real 1mm/5mm pairs improved LAA-950
CCC from 0.47 to 0.84 while the only weight that moved beyond float32 noise was
the output bias (-12.2 HU).

This module splits those decisions across solvers that each own ONE separately
falsifiable estimand and exchange a typed, machine-checkable artifact:

  OperatorSolver     (no learned parameters) -> (w_hat, sigma_hat, rho)
  the SCTE_R network (learned)               -> x_hat, conditioned on (w_hat, sigma_hat)
  DensitometrySolver (no learned parameters) -> (delta_hat, rho_struct, s)
  Arbiter                                    -> certified | flagged(term) | abstain

The load-bearing property is that a normalised slice-sensitivity kernel followed
by average pooling PRESERVES SLAB MEANS for every width, so the mean direction of
the data-consistency residual identifies a global attenuation displacement
independently of the width estimate — but only where the averaged support lies
inside the averaging set, which is why delta_hat is taken over an ERODED lung
interior (measured on a real pair: un-eroded, a perfect reconstruction scores
+6.9 HU and swings 9.4 HU across w in [4,10] mm; eroded, -0.55 HU and 0.04 HU).

The tail-recovery term `s` is EMPIRICAL and protocol-scoped: the closed-form
attenuation factor does not relate the observed thick lung variance to the thin
one (measured: off by 10-18x), so `s` is defined against a calibration fitted on
paired training data (scripts/fit_tail_calibration.py) and reports UNAVAILABLE
when no calibration exists for the current protocol.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from .forward_operator import SSPForwardOperator
from .model import SCTE_R

HU_SCALE = 1000.0            # volumes are carried in kHU
HU_AIR_FLOOR = -0.990        # kHU: below this is clipped air (outside body / airway lumen)
HU_LUNG_CEILING = -0.500     # kHU: conventional parenchyma window ceiling


# --------------------------------------------------------------------------
def lung_set(v):
    """Parenchyma set on whatever grid `v` lives, excluding clipped air."""
    return (v > HU_AIR_FLOOR) & (v < HU_LUNG_CEILING)


def eroded_lung_set(v, e_mask=8, e_z=1):
    """Interior of the lung set: no voxel whose averaged support crosses the
    boundary. Pure torch (max-pool erosion) so it runs on GPU inside training."""
    m = lung_set(v).float()
    if e_z > 0 or e_mask > 0:
        pad = (e_mask, e_mask, e_mask, e_mask, e_z, e_z)
        m = 1.0 - torch.nn.functional.max_pool3d(
            torch.nn.functional.pad(1.0 - m, pad, mode="constant", value=1.0),
            kernel_size=(2 * e_z + 1, 2 * e_mask + 1, 2 * e_mask + 1), stride=1)
    return m > 0.5


def z_variance(v, mask):
    """Through-plane variance component: var of first z-differences, halved."""
    d = v[..., 1:, :, :] - v[..., :-1, :, :]
    m = mask[..., 1:, :, :] & mask[..., :-1, :, :]
    if m.sum() < 8:
        return torch.tensor(float("nan"), device=v.device)
    return d[m].var() / 2.0


# --------------------------------------------------------------------------
class OperatorSolver(nn.Module):
    """Identify the through-plane acquisition, and measure the per-scan misfit.

    MEASURED IDENTIFIABILITY LIMIT (why this is protocol-level, not per-scan).
    Fitting w by re-blurring an estimate that was itself derived from y is
    circular and degenerate: on 12 real RPLHR pairs the fit collapses to the grid
    floor (2.00 mm) both when the reference is the trilinear up-sampling of y and
    when it is the model's own reconstruction, against a gold value of 6.00-6.50 mm
    obtained with the TRUE thin volume. A reference-free surrogate (the ratio of
    through-plane to in-plane texture of the thick series) is monotone in w on the
    simulated arm but only to ~2.2 mm precision, and its real-arm values fall
    outside the simulated range entirely. So:

      * `source='paired'`  - the offline fit, run ONCE per protocol on a handful of
        real thin/thick pairs (scripts/fit_protocol_operator.py). The fitted width
        is stable across cases (sd 0.23 mm over 50 pairs).
      * `source='protocol'` (DEFAULT at inference) - w is the protocol constant;
        the solver still returns a PER-SCAN misfit rho, i.e. how far this scan
        departs from the protocol it is claimed to belong to.

    This costs less than it looks, because the certificate's load-bearing term
    (delta_hat) is invariant to w by construction — measured spread 0.04 HU across
    w in [4,10] mm — so a protocol-level width with residual error does not
    contaminate it. `source='recon'` / `'upsampled'` remain available to reproduce
    the degenerate behaviour above.
    """

    def __init__(self, downsample=5, grid_mm=(2.0, 12.0, 0.25), source="protocol",
                 protocol_w=None):
        super().__init__()
        self.r = downsample
        self.grid = torch.arange(grid_mm[0], grid_mm[1] + 1e-9, grid_mm[2])
        self.source = source
        self.protocol_w = protocol_w

    def _misfit(self, x_ref, y, w):
        op = SSPForwardOperator(slice_fwhm_mm=float(w), downsample=self.r)
        res = op(x_ref) - y
        return res.norm() / (y - y.mean()).norm().clamp_min(1e-8)

    @torch.no_grad()
    def forward(self, y_thick, x_ref=None):
        """y_thick (B,1,Dt,H,W) in kHU -> (w_hat[mm], sigma_hat[kHU], rho, at_edge)."""
        B, _, Dt, H, W = y_thick.shape
        D = Dt * self.r
        dz = y_thick[..., 1:, :, :] - y_thick[..., :-1, :, :]
        sigma_hat = dz.flatten(1).std(dim=1) / math.sqrt(2)

        if self.source == "protocol":
            if self.protocol_w is None:
                raise ValueError("source='protocol' needs a fitted protocol width "
                                 "(scripts/fit_protocol_operator.py)")
            w_hat = torch.full((B,), float(self.protocol_w), device=y_thick.device)
            rho = torch.zeros(B, device=y_thick.device)
            if x_ref is not None:                    # per-scan misfit under that width
                for b in range(B):
                    rho[b] = self._misfit(x_ref[b:b + 1], y_thick[b:b + 1], self.protocol_w)
            return w_hat, sigma_hat, rho, torch.zeros(B, dtype=torch.bool,
                                                      device=y_thick.device)

        if x_ref is None or self.source == "upsampled":
            x_ref = SSPForwardOperator(downsample=self.r).upsample_to_grid(y_thick, D)
        w_hat = torch.empty(B, device=y_thick.device)
        rho = torch.empty(B, device=y_thick.device)
        for b in range(B):
            errs = [float(((SSPForwardOperator(slice_fwhm_mm=float(w), downsample=self.r)
                            (x_ref[b:b + 1]) - y_thick[b:b + 1]) ** 2).mean())
                    for w in self.grid]
            k = int(np.argmin(errs))
            w_hat[b] = float(self.grid[k])
            rho[b] = self._misfit(x_ref[b:b + 1], y_thick[b:b + 1], float(self.grid[k]))
        at_edge = (w_hat <= float(self.grid[0]) + 1e-6) | (w_hat >= float(self.grid[-1]) - 1e-6)
        return w_hat, sigma_hat, rho, at_edge


# --------------------------------------------------------------------------
class TailCalibration:
    """Protocol-scoped relation predicting the thin-slice through-plane lung
    variance from quantities observable at inference. Fitted by
    scripts/fit_tail_calibration.py; absent -> the tail term is UNAVAILABLE."""

    FEATURES = ("var_z_thick", "w_hat", "sigma_hat", "var_z_thick_x_w")

    def __init__(self, coef=None, intercept=0.0, sigma_v=None, protocol=None,
                 informative=False, protocol_w=None):
        self.coef, self.intercept = coef, intercept
        self.sigma_v, self.protocol, self.informative = sigma_v, protocol, informative
        # the protocol's fitted effective slice width travels with the calibration:
        # both are protocol-scoped and both must be refitted together
        self.protocol_w = protocol_w

    @classmethod
    def load(cls, path):
        if path is None or not os.path.exists(path):
            return cls()
        d = json.load(open(path))
        coef = np.asarray(d["coef"], dtype=float) if d.get("coef") is not None else None
        return cls(coef, float(d.get("intercept", 0.0)),
                   d.get("sigma_v"), d.get("protocol"),
                   bool(d.get("informative", False)), d.get("protocol_w"))

    @property
    def available(self):
        return self.coef is not None and self.informative

    def predict(self, var_z_thick, w_hat, sigma_hat):
        x = np.array([var_z_thick, w_hat, sigma_hat, var_z_thick * w_hat], dtype=float)
        return float(self.intercept + self.coef @ x)


# --------------------------------------------------------------------------
class DensitometrySolver(nn.Module):
    """Turn the disagreement between the reconstruction and the observation into
    named quantities: a displacement term, a structural remainder, and (when a
    protocol calibration exists) a tail-recovery term.

    delta_hat is differentiable — it is the one term that feeds training."""

    def __init__(self, downsample=5, e_mask=8, e_z=1, calibration=None, learned_op=None):
        super().__init__()
        self.r = downsample
        # when a fitted operator is supplied it REPLACES the Gaussian in every place
        # the certificate applies the physics; it stays mean-preserving by construction
        # (softmax basis + zero-mean correction), so delta_hat's invariance survives
        self.learned_op = learned_op
        self.e_mask, self.e_z = e_mask, e_z
        self.cal = calibration or TailCalibration()

    def forward(self, x_hat, y_thick, w_hat, lung_thick=None):
        """`lung_thick` (B,1,Dt,H,W) bool: an ANATOMICAL lung mask already pooled onto
        the thick grid. When given it replaces the HU-window set — measured on real
        cases, the HU window gives LAA-950 ~10% where the lobe mask gives ~1%, because
        its denominator is the (-990,-500) band rather than the parenchyma."""
        B = y_thick.shape[0]
        dev = y_thick.device
        delta = torch.zeros(B, device=dev)
        rho_struct = torch.zeros(B, device=dev)
        s = [float("nan")] * B
        abstain = torch.zeros(B, dtype=torch.bool, device=dev)
        if lung_thick is None:
            Le = eroded_lung_set(y_thick, self.e_mask, self.e_z)
        else:
            m = lung_thick.float()
            pad = (self.e_mask, self.e_mask, self.e_mask, self.e_mask, self.e_z, self.e_z)
            Le = (1.0 - torch.nn.functional.max_pool3d(
                torch.nn.functional.pad(1.0 - m, pad, mode="constant", value=1.0),
                kernel_size=(2 * self.e_z + 1, 2 * self.e_mask + 1, 2 * self.e_mask + 1),
                stride=1)) > 0.5
        for b in range(B):
            op = (self.learned_op if self.learned_op is not None
                  else SSPForwardOperator(slice_fwhm_mm=float(w_hat[b]), downsample=self.r))
            R = op(x_hat[b:b + 1]) - y_thick[b:b + 1]
            m = Le[b:b + 1]
            if m.sum() < 8:
                abstain[b] = True
                delta[b] = R.mean()                       # keep it finite/differentiable
            else:
                delta[b] = R[m].mean()
            denom = (y_thick[b:b + 1] - y_thick[b:b + 1].mean()).norm().clamp_min(1e-8)
            rho_struct[b] = (R - delta[b]).norm() / denom
            if self.cal.available:
                yb = y_thick[b:b + 1]
                vzy = float(z_variance(yb, lung_set(yb)))
                vzx = float(z_variance(x_hat[b:b + 1].detach(),
                                       lung_set(x_hat[b:b + 1].detach())))
                dz = yb[..., 1:, :, :] - yb[..., :-1, :, :]
                sig = float(dz.std()) / math.sqrt(2)      # same feature the fit used
                if not (math.isnan(vzy) or math.isnan(vzx)):
                    v_hat = self.cal.predict(vzy, float(w_hat[b]), sig)
                    s[b] = (v_hat - vzx) / self.cal.sigma_v
        return dict(delta_hat=delta, rho_struct=rho_struct, s=s, abstain=abstain)


# --------------------------------------------------------------------------
class Arbiter:
    """Feasibility verdict on the certificate — not a score.

    Thresholds are in the certificate's own units: tau_delta in HU, tau_rho as a
    fraction of the observation's own variation, tau_s in units of the
    calibration's held-out residual spread."""

    def __init__(self, tau_delta=2.0, tau_rho=0.15, tau_s=2.0):
        self.tau_delta, self.tau_rho, self.tau_s = tau_delta, tau_rho, tau_s

    def __call__(self, cert):
        out = []
        for i in range(len(cert["delta_hat"])):
            if bool(cert["abstain"][i]):
                out.append(("abstain", "empty_lung_interior")); continue
            d = abs(float(cert["delta_hat"][i]) * HU_SCALE)
            r = float(cert["rho_struct"][i])
            s = cert["s"][i]
            viol = []
            if d > self.tau_delta:
                viol.append(("displacement", d / self.tau_delta))
            if r > self.tau_rho:
                viol.append(("structural", r / self.tau_rho))
            if s == s and abs(s) > self.tau_s:       # s == s filters NaN/UNAVAILABLE
                viol.append(("tail_recovery", abs(s) / self.tau_s))
            if not viol:
                out.append(("certified", None))
            else:
                out.append(("flagged", max(viol, key=lambda t: t[1])[0]))
        return out


# --------------------------------------------------------------------------
class ARC_CT(nn.Module):
    """SCTE_R with the acquisition identified per scan and the reconstruction
    accompanied by a certificate.

    Warm-starts from a single-pass checkpoint: `load_state_dict(..., strict=False)`
    is safe because the added solvers carry no learned parameters."""

    def __init__(self, downsample=5, slice_fwhm_mm=5.0, base_ch=32, n_blocks=6,
                 cond_dim=32, dc_steps=1, n_kernels=16, e_mask=8, e_z=1,
                 op_source="protocol", calibration=None, identify=True,
                 grid_mm=(2.0, 12.0, 0.25), protocol_w=None, learned_op=None):
        super().__init__()
        self.net = SCTE_R(downsample=downsample, slice_fwhm_mm=slice_fwhm_mm,
                          base_ch=base_ch, n_blocks=n_blocks, cond_dim=cond_dim,
                          dc_steps=dc_steps, n_kernels=n_kernels)
        self.op = self.net.op                       # so evaluate.py keeps working
        self.downsample = downsample
        self.nominal_fwhm = slice_fwhm_mm
        self.identify = identify                    # False = ablation (a): nominal width
        if protocol_w is None and calibration is not None:
            protocol_w = getattr(calibration, "protocol_w", None)
        self.protocol_w = protocol_w if protocol_w is not None else slice_fwhm_mm
        self.operator = OperatorSolver(downsample, grid_mm, op_source,
                                       protocol_w=self.protocol_w)
        self.densitometry = DensitometrySolver(downsample, e_mask, e_z, calibration,
                                               learned_op=learned_op)
        if learned_op is not None:      # data consistency also uses the fitted physics
            self.net.m3.op = learned_op
            self.op = learned_op

    def forward(self, y_thick, kernel_id=None):
        B = y_thick.shape[0]
        if self.identify:
            w_hat, sigma_hat, rho, at_edge = self.operator(y_thick)
        else:
            w_hat = torch.full((B,), self.nominal_fwhm, device=y_thick.device)
            sigma_hat = torch.zeros(B, device=y_thick.device)
            rho = torch.zeros(B, device=y_thick.device)
            at_edge = torch.zeros(B, dtype=torch.bool, device=y_thick.device)

        # reconstruct under the identified acquisition (M1 takes per-sample widths)
        D = y_thick.shape[2] * self.downsample
        cond, w0, sigma = self.net.m1(y_thick, kernel_id, w0_nominal_fwhm=w_hat)
        y_up = self.net.op.upsample_to_grid(y_thick, D)
        x_hat = self.net.m2(y_up, cond)
        x_hat = self.net.m3(x_hat, y_thick, sigma_vox=w0)
        dc_res = self.net.m3.residual(x_hat, y_thick, sigma_vox=w0)
        rel = self.net.m5(dc_res)

        cert = self.densitometry(x_hat, y_thick, w_hat)
        return dict(x_hat=x_hat, cond=cond, w0=w0, noise_sigma=sigma,
                    dc_residual=dc_res, reliability=rel,
                    w_hat=w_hat, sigma_hat=sigma_hat, rho=rho, fit_at_edge=at_edge,
                    **cert)


# --------------------------------------------------------------------------
class SCTER_Iterative(ARC_CT):
    """The iterative variant the packaged `build_model(iters>1)` switch expects.

    Rounds re-identify the acquisition against the CURRENT reconstruction and
    re-run the network on the data-consistent estimate; the certificate's
    displacement term is the early-stop signal (a round that no longer reduces
    |delta_hat| is not taken)."""

    def __init__(self, iters=2, **kw):
        kw.setdefault("op_source", "recon")
        super().__init__(**kw)
        self.iters = int(iters)

    def forward(self, y_thick, kernel_id=None):
        out = super().forward(y_thick, kernel_id)
        for _ in range(max(0, self.iters - 1)):
            w_hat, sigma_hat, rho, at_edge = self.operator(y_thick, out["x_hat"].detach())
            cond, w0, sigma = self.net.m1(y_thick, kernel_id, w0_nominal_fwhm=w_hat)
            x_hat = self.net.m2(self.net.op.upsample_to_grid(
                y_thick, y_thick.shape[2] * self.downsample), cond)
            x_hat = self.net.m3(x_hat, y_thick, sigma_vox=w0)
            cert = self.densitometry(x_hat, y_thick, w_hat)
            improved = cert["delta_hat"].abs() < out["delta_hat"].abs()
            if not bool(improved.any()):
                break
            keep = improved.view(-1, 1, 1, 1, 1)
            out = dict(out)
            out["x_hat"] = torch.where(keep, x_hat, out["x_hat"])
            out["w_hat"] = torch.where(improved, w_hat, out["w_hat"])
            out["rho"] = torch.where(improved, rho, out["rho"])
            out["delta_hat"] = torch.where(improved, cert["delta_hat"], out["delta_hat"])
            out["rho_struct"] = torch.where(improved, cert["rho_struct"], out["rho_struct"])
            dc_res = self.net.m3.residual(out["x_hat"], y_thick, sigma_vox=w0)
            out["dc_residual"], out["reliability"] = dc_res, self.net.m5(dc_res)
        return out
