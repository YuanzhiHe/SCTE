# Addendum — what the implementation measured, and what it changed in the card

The idea card (`phase4/idea.*.md`) was written before ARC-CT was built. Building it
on the real RPLHR-CT pairs falsified one of its claims and rescaled two others.
This file records the corrections so the card and the code do not drift apart; the
card's own audit trail (Phase 2.3 coherence gate, Phase 3.2 audit) already carries
the two defects found *before* implementation.

## 1. Per-scan operator identification is NOT possible — the claim is now protocol-level

The card's audit leg said the acquisition is identified **per scan** by fitting the
effective slice width against the observation. Measured on 12 real pairs:

| reference the fit runs against | fitted width |
|---|---|
| the TRUE thin volume (gold, unavailable at inference) | 6.00 – 6.50 mm |
| the trilinear up-sampling of the thick series | **2.00 mm — the grid floor** |
| the model's own reconstruction | **2.00 mm — the grid floor** |

Both inference-time references are derived from `y` itself and are already smooth
along z, so the best-fitting extra blur is none: the fit is **circular and
degenerate**, not merely biased. A reference-free surrogate (the ratio of
through-plane to in-plane texture of the thick series) is monotone in `w` on the
simulated arm but only to ~2.2 mm precision (dratio/dw = -0.16/mm against a
per-case spread of 0.35), and its real-arm values (3.96–6.34) fall entirely outside
the simulated range (2.39–3.58), so it cannot be transported either.

**Correction.** The width is fitted ONCE PER PROTOCOL on a handful of real pairs
(`scripts/fit_protocol_operator.py`): over 85 RPLHR training pairs the fit is
6.25 mm median, 6.16 ± 0.51 mm — stable enough to be a protocol constant. What
stays per-scan is the MISFIT `rho` under that constant, i.e. how far this scan
departs from the protocol it is claimed to belong to.

**Why the method survives this.** The certificate's load-bearing term `delta_hat`
is invariant to `w` by construction (measured spread 0.04 HU across w in
[4,10] mm), so a protocol-level width with residual error does not contaminate it.
The audit leg's claim becomes: *the operator must be identified from data rather
than assumed, and the certificate must be invariant to the residual error in that
identification* — which is the property we can actually demonstrate.

## 2. The certificate's thresholds are calibrated against a known-perfect reconstruction

The card fixed absolute thresholds (tau_delta = 2 HU, tau_rho = 0.15,
tau_s = 0.15). Running step S6 (the certificate with `x_hat := the TRUE thin
volume`) showed both `rho_struct` and `s` are patch-geometry dependent, so absolute
constants condemn ground truth. Thresholds are now the null distribution's
`|mean| + 2 sd`, measured with a perfect reconstruction on TRAINING pairs only:

| protocol | tau_delta | tau_rho | tau_s |
|---|---|---|---|
| RPLHR-CT real 5 mm (85 train pairs) | 4.33 HU | 0.774 | 2.01 sigma_v |
| SIMULATED 8 mm (85 train volumes) | 0.29 HU | 0.224 | n/a (uncalibrated) |

The 15x difference in `tau_delta` between the two rows is itself a measurement:
it is the sim-to-real gap expressed in the certificate's own units.

Sanity check on the held-out 50 test pairs with the train-fitted thresholds: a
perfect reconstruction is certified in **45/50** cases (delta_hat +1.04 ± 1.44 HU,
s -0.45 ± 1.03 sigma_v), i.e. a ~10% false-alarm rate that is reported, not hidden.

## 3. The tail-recovery term needed the calibration's own feature vector

`s` was reported at -2.5 sigma_v for a perfect reconstruction until the solver was
fixed to pass the same `sigma_hat` feature the calibration was fitted with; it is
now +0.03 ± 0.99 sigma_v on training pairs, i.e. correctly centred. The relation
itself is weak but informative (held-out residual sd 234.5 HU^2 against a
between-case sd of 271.0 HU^2) and is shipped with an explicit `informative` flag —
when it is false, ARC-CT reports the term UNAVAILABLE instead of shipping a
calibration that carries no information.

## 4. The displacement penalty has to be priced in HU

Carried in kHU (the volume's internal unit) the penalty is ~1000x too small to
compete with the trajectory guardrail and silently does nothing: measured
`delta_hat` = -16.6 HU with lambda_delta = 1.0 versus -17.6 HU without the penalty
at all. Priced in HU it takes effect immediately (mean |delta_hat| drops from
~17 HU to ~0.9 HU within one epoch at lambda_delta = 0.1). This is the same
unit-scale failure that the packaged code had in its noise model — worth stating in
the paper as a reproducibility note rather than a footnote.

## 5. Author decisions, resolved

1. **What the operator fit runs against** — resolved by measurement, not preference:
   neither inference-time reference works (see §1), so the fit is offline and
   protocol-level, against the true thin volume of the calibration pairs.
2. **What a protocol is** — the tuple (manufacturer/model, reconstruction kernel,
   nominal thickness, r); with metadata missing it degrades to (r, quantised w).
   Stored in the calibration file and echoed in every certificate.
3. **How many real pairs the bias-only control may use** — reported as a curve over
   N in {1, 3, 5, 10, 25} rather than a single number (see the results table).
4. **8 mm scope** — restricted to the SIMULATED arm, as instructed: no public real
   8 mm paired data exists, and the claim is labelled accordingly everywhere.
