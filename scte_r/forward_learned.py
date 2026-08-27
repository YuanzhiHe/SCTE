"""Learned forward operator:  A_theta(x) = sum_k a_k A_{w_k}(x) + zero-mean correction.

Why. Everything that tried to USE the forward model as a hard constraint failed:
per-step measurement guidance drove LAA-950 from 0.34 to 4.95 pp, and extra
data-consistency projection steps drove it from 0.36 to 1.48 pp — while making the
certificate's terms look better. The cause is measurable: a single Gaussian slice
profile leaves ~12.8% of the real thick series unexplained, so constraining the
solution to satisfy it exactly pulls the solution away from the truth. The fix is
not a better sampler, it is a better operator.

The forward problem is WELL-POSED (1 mm -> 5 mm is a deterministic map, and 85 real
pairs are plenty), unlike the inverse problem, so the operator can simply be fitted.

Two structural constraints keep the certificate valid:

  * MEAN PRESERVATION. The basis coefficients are softmax-normalised and every
    A_{w_k} is itself mean-preserving, so their convex combination is too; and the
    learned correction is made zero-mean per slab by construction. delta_hat's
    invariance to the operator — the property the whole certificate rests on —
    therefore survives learning. A free-form learned operator would destroy it.
  * The basis is a set of Gaussian widths, so the fitted `a_k` are directly
    interpretable as a non-parametric slice-sensitivity profile: the paper can
    report the real SSP shape rather than assert a Gaussian.
"""
import torch
import torch.nn as nn

from .forward_operator import SSPForwardOperator


class LearnedOperator(nn.Module):
    def __init__(self, downsample=5, widths=(3., 4., 5., 6., 7., 8., 10., 12.),
                 base_ch=16, correction=True):
        super().__init__()
        self.r = downsample
        self.widths = tuple(widths)
        self.ops = [SSPForwardOperator(slice_fwhm_mm=w, downsample=downsample)
                    for w in self.widths]
        # start at the nominal width so training begins from the current physics
        init = torch.full((len(self.widths),), -2.0)
        init[min(range(len(self.widths)), key=lambda i: abs(self.widths[i] - 5.0))] = 2.0
        self.logits = nn.Parameter(init)
        self.correction = correction
        if correction:
            self.net = nn.Sequential(
                nn.Conv3d(1, base_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(base_ch, base_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(base_ch, 1, 3, padding=1))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    @property
    def coeffs(self):
        return torch.softmax(self.logits, 0)

    def effective_fwhm(self):
        """The basis-weighted width, for reporting alongside the fitted profile."""
        a = self.coeffs.detach()
        return float((a * torch.tensor(self.widths, device=a.device)).sum())

    def forward(self, x, sigma_vox=None):        # signature matches SSPForwardOperator
        a = self.coeffs
        base = sum(a[k] * self.ops[k](x) for k in range(len(self.ops)))
        if not self.correction:
            return base
        c = self.net(base)
        c = c - c.mean(dim=(2, 3, 4), keepdim=True)      # zero-mean: keeps delta_hat valid
        return base + c

    def upsample_to_grid(self, y_thick, out_depth):
        return self.ops[0].upsample_to_grid(y_thick, out_depth)
