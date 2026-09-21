# Project Truth

**Active manuscript**: `output/doc/manuscript.md`
**Venue**: npj Digital Medicine, Article
**Contribution type**: mixed, dominant = benchmark/analysis with an enabling method

## Central claim

Image-quality gains from thick-to-thin CT restoration do not transfer to emphysema
densitometry. The reason is measurable: on this inverse problem the optimum of a
deterministic regression is a constant global HU displacement, and threshold
statistics are maximally sensitive to exactly that degree of freedom. A residual-space
flow-matching sampler removes the displacement and recovers 95% of the thick-slice
bias, at a PSNR below that of the unreconstructed 5 mm series.

## What the paper is NOT claiming

- Not claiming image quality superiority. PSNR is lower than CTHNet by 3.96 dB and
  lower than the 5 mm series by 0.78 dB. This is a stated trade, not a hidden one.
- Not claiming a per-scan validity guarantee. The certificate is a site-level
  deployment gate; within a domain its verdict carries no case-level information.
- Not claiming reader-level diagnostic benefit yet. No reading study has been run.

## Stable constraints

- Private cohorts: only derived per-case CSV/log numbers leave the hospital.
  `ID_MAP_DO_NOT_EXPORT.csv` and images stay on the secure machine.
- Training cohort (203 cases) and validation cohort (494 = 50 calibration + 444
  reported) are from DIFFERENT hospitals. This is external validation, not an
  internal split. Some intermediate documents called both by one hospital name.
- The prior PPT from an earlier attempt must not be cited or its method copied.
- Figures: colour in panels, black running text.

## Blocking dependency

The two target papers could not be retrieved automatically (nature.com returns a
303 to an auth endpoint for every route, including the gold-OA PDF):
- Ren et al., npj Digit Med 2024, `10.1038/s41746-024-01338-8` (CTHNet; our backbone
  and baseline)
- Yu et al., npj Digit Med 2026, `10.1038/s41746-026-03253-6` — "Benchmarking
  AI-generated thin-slice CT under clinical reconstruction conditions: a multicohort
  study". Adjacent and possibly competing; must be read before Claim 1 is finalised.

Both are needed in `input/` before structural mirroring and positioning are final.
