"""Composite SCTE-R training loss.

  L = L_voxel                                  (image fidelity)
    + w_dc   * L_dataconsistency               (M3, HU/mass conservation)
    + w_traj * L_functional_trajectory         (M4 guardrail, LAA/Perc15 vs width)
    + w_nps  * L_noise_calibration             (M4 guardrail, NPS/noise match)

Trajectory + NPS terms are the quantitative-fidelity guardrails that separate
SCTE-R from a pure image-fidelity DLS.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import laa_soft, perc15, percN, nps_1d_z
from .forward_operator import SSPForwardOperator


class SCTERLoss(nn.Module):
    def __init__(self, operator: SSPForwardOperator, w_dc=1.0, w_traj=0.5,
                 w_nps=0.1, ladder=(0.0, 1.0, 2.0, 3.0), w_delta=0.0,
                 w_quant=0.0, quantiles=(0.01, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.90),
                 w_bio=0.0):
        super().__init__()
        self.op = operator
        self.w_dc, self.w_traj, self.w_nps = w_dc, w_traj, w_nps
        self.ladder = ladder
        # M4b (ARC-CT): price the one direction the trajectory guardrail can
        # otherwise use to buy densitometric accuracy for free — a global HU
        # displacement of the reconstruction. Only active when the model emits
        # `delta_hat` (scte_r/agentic.py).
        self.w_delta = w_delta
        # M4c: distribution matching on the lung-masked attenuation histogram.
        # The LAA surrogate is a THRESHOLD functional, so a global shift satisfies
        # it without fixing the histogram (measured: the packaged objective buys its
        # whole LAA gain with a -15 HU shift). Matching a set of quantiles cannot be
        # satisfied that way unless the shape already matches, because a shift moves
        # every quantile by the same amount.
        self.w_quant = w_quant
        # Biomarker multi-task term: constrain the reported emphysema indices
        # DIRECTLY, at native resolution, with no self-coarsening ladder. This is
        # the published recipe for "optimise the clinical index, not the pixel";
        # it is included so the displacement it induces can be measured rather
        # than argued about.
        self.w_bio = w_bio
        self.quantiles = tuple(quantiles)

    def functional_trajectory(self, x_hat, x_true):
        """Match LAA/Perc15 vs added-blur-width curve (self-coarsening ladder)."""
        loss = 0.0
        for add in self.ladder:
            if add > 0:
                xh = self.op.blur_z(x_hat, add)
                xt = self.op.blur_z(x_true, add)
            else:
                xh, xt = x_hat, x_true
            loss = loss + F.l1_loss(laa_soft(xh), laa_soft(xt))
            loss = loss + 0.01 * F.l1_loss(perc15(xh), perc15(xt))
        return loss / len(self.ladder)

    def biomarker(self, x_hat, x_true):
        """LAA-950 + LAA-910 + Perc15/10/5 matched at native resolution."""
        l = F.l1_loss(laa_soft(x_hat), laa_soft(x_true))
        l = l + F.l1_loss(laa_soft(x_hat, thresh=-0.91), laa_soft(x_true, thresh=-0.91))
        for q in (0.15, 0.10, 0.05):                      # HU-scale terms, priced down
            l = l + 0.01 * F.l1_loss(percN(x_hat, q), percN(x_true, q))
        return l

    def quantile_match(self, x_hat, x_true):
        """1-D distributional distance over the lung-window voxels, in kHU."""
        m = (x_true > -1.0) & (x_true < -0.5)
        if m.sum() < 64:
            return x_hat.sum() * 0.0
        a, b = x_hat[m].float(), x_true[m].float()
        q = torch.tensor(self.quantiles, device=a.device, dtype=a.dtype)
        return (torch.quantile(a, q) - torch.quantile(b, q)).abs().mean()

    def noise_calibration(self, x_hat, x_true):
        """Match the through-plane NPS magnitude to the real thin slice."""
        return F.l1_loss(nps_1d_z(x_hat), nps_1d_z(x_true))

    def forward(self, out, x_true, y_thick):
        x_hat = out["x_hat"]
        l_vox = F.l1_loss(x_hat, x_true)
        l_dc = F.mse_loss(self.op(x_hat, out.get("w0")), y_thick)
        l_traj = self.functional_trajectory(x_hat, x_true)
        l_nps = self.noise_calibration(x_hat, x_true)
        total = l_vox + self.w_dc * l_dc + self.w_traj * l_traj + self.w_nps * l_nps
        logs = dict(voxel=l_vox.item(), dc=l_dc.item(),
                    traj=l_traj.item(), nps=l_nps.item())
        if self.w_delta and out.get("delta_hat") is not None:
            # priced in HU, like the trajectory guardrail is priced in percentage
            # points — in kHU the term is ~1000x too small to compete and the
            # penalty silently does nothing (measured: -16.6 HU vs -17.6 HU without it)
            l_delta = 1000.0 * out["delta_hat"].abs().mean()
            total = total + self.w_delta * l_delta
            logs["delta"] = l_delta.item()                    # HU
        if self.w_bio:
            l_bio = self.biomarker(out["x_hat"], x_true)
            logs["bio"] = float(l_bio)
            total = total + self.w_bio * l_bio

        if self.w_quant:
            l_q = self.quantile_match(x_hat, x_true)
            total = total + self.w_quant * 1000.0 * l_q      # priced in HU
            logs["quant"] = 1000.0 * l_q.item()
        logs["total"] = total.item()
        return total, logs
