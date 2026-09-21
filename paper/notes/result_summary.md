# Result Summary

Numbers below are final unless marked. Cohort labels: EXT = 444-case external
validation cohort (two manufacturers, two kernels); PUB = 50-case public paired
cohort; both whole-volume, lung set from TotalSegmentator lobe masks.

## Supported, figure-anchored

| Finding | Value | Anchor |
|---|---|---|
| 5 mm read directly (EXT) | LAA-950 2.83% vs truth 7.82%; bias −4.99 pp, LoA −13.50 to +3.51, CCC 0.631 | Table 1, Fig 1d |
| CTHNet (EXT) | bias −4.48 pp, CCC 0.682, PSNR 30.77 | Table 1 |
| Lanczos (EXT) | bias −4.41 pp, CCC 0.689, PSNR 28.79 | Table 1 |
| Ours zero-shot (EXT) | bias −0.26 pp, LoA −5.61 to +5.08, CCC 0.933, PSNR 26.81 | Table 1, Fig 1e |
| Fraction of thick-slice bias recovered | CTHNet 10%, Lanczos 12%, ours 95% | Fig 1d |
| Perc15 bias (EXT) | 5 mm +25.7, Lanczos +23.5, CTHNet +19.3, ours +2.5 HU | Table 3 |
| Displacement of deterministic regression (EXT) | −16.32 ± 0.65 HU, n=444 | Fig 1f, Table 5 |
| Bias-only control | −39.8 to −41.8 HU, 0/444 certified | Table 5 |
| Public benchmark (PUB) | CTHNet/TVSRN/I3Net bias −0.82/−0.90/−0.86 vs Lanczos −0.78 pp | Table 4, Fig 2a |
| Reference burden, PUB | LAA-950 1.21 ± 1.14% — a near-normal cohort | Table 4 |
| Five-lobe agreement | CCC 0.929–0.956 | Table 6 |
| Certificate by domain | 80% / 16.1% / 4.8% / ~99% | Fig 1g |
| Conservative lung-mask variant | MAE 5.05 → 5.02 pp | text |

## Supported, secondary

- Integration steps 32 vs 64: paired |error| difference 95% CI [−0.06, +0.46] pp,
  lung PSNR +0.09 dB (n=20). Robustness note only.
- Nodule contrast: 5 mm retains 47%, loses 31% of sub-6 mm nodules; reconstruction
  recovers 70% zero-shot.

## Weak or excluded

- Site adaptation: better LAA-950 CCC (0.948) and PSNR (+1.5 dB), worse LAA-910 on
  every statistic and Perc15/Perc10 MAE +22%. Report as an upper bound with local
  paired data, not as the main arm; the deployment target cannot produce such data.
- Per-scanner calibration: no effect. Both scanners fit the same effective width.
  Not reported.
- Certificate case-level power: within SIEMENS, certified 0/50 vs flagged 5/261
  flips at the 10% cut-off, Fisher p = 1.0. Scope the claim to site level.
- Hybrid arm with the public checkpoint: configuration mismatch (checkpoint trained
  without `--base_recon`); lowest PSNR and SSIM of all arms. Excluded.

## Sample sizes

n = 444 for per-arm descriptive statistics. n = 438 for paired comparisons: six
cases excluded because some arm produced a non-finite value. Four of those six came
from the model diverging on cases whose lung masks were normal — a limitation to
state, now handled by abstaining rather than certifying.
