# Literature search — thick-slice CT restoration and emphysema densitometry

Purpose: novelty grounding for three claims before a clinical-imaging submission.
Mode: standard. Date: 2026-09-06.

Source policy applied: primary/publisher pages preferred; MDPI and other policy-excluded
venues omitted from the scored table (see search-notes.md).

## Cluster A — slice thickness and reconstruction parameters change emphysema indices
**Status: established. Cite, do not claim.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| Quantitative Emphysema Measurement On Ultra-High-Resolution CT Scans | Int J COPD 2019 | method+benchmark | LAV% 8.9 at 0.25 mm vs 7.3 at 0.5 mm, same −950 HU threshold |
| Effect of slice thickness on quantitative analysis of interstitial lung disease | 2025 | benchmark | Thickness changes emphysema/consolidation; GGO and total ILD extent unaffected |
| The Impact of CT Reconstruction Parameters on Emphysema Index Quantification | 2025 | benchmark | Emphysema volume 766.9 vs 482.6 mL (HRCT vs conventional), partial-volume driven |

## Cluster B — thick→thin super-resolution, and how it is evaluated
**Status: established practice. Our baselines.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| Spatial resolution enhancement using deep learning improves chest disease diagnosis based on thick slice CT (CTHNet) | npj Digital Medicine 2024 | method+benchmark | Our backbone. Reports image quality and READER diagnosis, incl. nodule sensitivity — not densitometric agreement |
| RPLHR-CT Dataset and Transformer Baseline (TVSRN) | MICCAI 2022 | method+benchmark | The real-paired benchmark; introduces the pseudo-LR vs real-LR domain gap |
| I3Net: Inter-Intra-slice Interpolation Network | IEEE TMI 2024 | pure method | Third baseline |
| MedSR-Impact: Transformer-Based SR for Lung CT Segmentation, Radiomics, Classification, Prognosis | arXiv 2025 | method+benchmark | **Closest to claim 1's framing**: evaluates SR on downstream quantitative endpoints. Reports IMPROVEMENT (+4% Dice, higher radiomic reproducibility, +0.06 C-index) |
| Improved Consistency of Lung Nodule Categorization with Heterogeneous Slice Thickness by 3D Super-Resolution | 2025 | method+benchmark | Nodule categorisation 72.7% -> 94.5% after SR |

## Cluster C — deep-learning reconstruction introduces bias in densitometry
**Status: established for DENOISING/dose-reduction DLR. Not shown for slice-thickness SR.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| The effect of deep learning reconstruction on abdominal CT densitometry and image quality | European Radiology 2022 | survey/meta-analysis | Global attenuation shifts exist and are handled by bias correction against a known region |
| Deep learning image reconstruction algorithm for quantitative assessment (COPD) | Int J COPD | method+benchmark | At ultra-low dose, IR and DLR both bias emphysema measurement; histogram shape driven by noise and filtering |

## Cluster D — reusing archived CT for quantification
**Status: same motivation as ours, different lever.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| Kernel Conversion for Robust Quantitative Measurements of Archived Chest CT Using Deep Learning Image-to-Image Translation | 2022 | method | Explicitly motivated by making ARCHIVED CT quantitatively usable — our deployment motivation, addressing kernel rather than thickness |

## Cluster E — hallucinations and the null space
**Status: theory exists; explains one of our negative results.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| On hallucinations in tomographic image reconstruction | IEEE TMI 2021 | theory | Decomposes the estimate into measurement- and null-space components; hallucinations live in the null space. **Predicts our finding that a data-consistency-based local certificate cannot localise fabrications** (AUC ≈ 0.5), and why sample spread can (AUC 0.919) |

## Cluster F — uncertainty quantification without ground truth
**Status: crowded and closest to claim 3. Must position against it.**

| Paper | Venue/Year | Type | Relevance |
|---|---|---|---|
| Self-supervised Conformal Prediction for UQ in Imaging Problems | 2025 | theory+method | Calibrates from noisy measurements via SURE, no ground truth; marginal coverage guarantees |
| Self-supervised conformal prediction for Poisson imaging problems | 2025 | theory+method | Same family, Poisson noise |
| Metric-Guided Conformal Bounds for Probabilistic Image Reconstruction | 2025 | method | Bounds on a downstream metric rather than pixels |
| Non-Reference Quality Assessment for Medical Imaging | arXiv 2024 | method | No-reference IQA; perceptual, not physical validity |
| Validation and Generalizability of Self-Supervised Image Reconstruction for Undersampled MRI | MELBA 2022 | benchmark | No-reference metrics track human ratings for generalisability |
