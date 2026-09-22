# SCTE-R: thick-slice CT restoration for emphysema densitometry

Restoring 1 mm CT from 5 mm reconstructions so that the emphysema indices
(LAA-950, LAA-910, Perc15/10/5) come out right, not only the pixels.

## The finding this is built on

Three published thick-to-thin networks, retrained on one split and scored under one
protocol (50 whole volumes, TotalSegmentator lobe masks):

| method | lung PSNR | LAA-950 MAE | **LAA-950 CCC** |
|---|---|---|---|
| Lanczos-3 interpolation | 27.65 | 0.78 | 0.259 |
| TVSRN (MICCAI 2022) | 31.08 | 0.90 | 0.068 |
| CTHNet (npj Digit. Med. 2024) | 31.60 | 0.82 | 0.152 |
| I3Net (IEEE TMI 2024) | 31.34 | 0.86 | 0.100 |

Each beats interpolation by 3.4 to 4.0 dB, and none improves the densitometry: their
LAA-950 concordance sits below plain interpolation. Image fidelity and density
fidelity are not the same objective.

The reason is a global attenuation displacement. Every deterministic regression we
trained settles at one regardless of the loss, and adding the biomarker term to the
objective moves it by 0.01 HU. On the public cohort the displacement is about 20 HU;
on a 444-examination external cohort from a second hospital it measures
−16.32 ± 0.65 HU, a spread too small for a patient-level effect.

## What this repo adds

- A reference-free indicator of whether a model is competent on a given scanner.
  The displacement `δ̂` is computable from the thick observation alone, because the
  forward operator passes a constant offset through unchanged. It separates a
  recovered distribution from a shifted one: displaced arms certify 0/50, ours 47/50,
  with `|δ̂|` inside the perfect-reconstruction null. A stratified audit on the
  external cohort places its resolution at the scanner, not the individual scan;
  within one scanner the verdict carries no case-level information
  (Fisher exact test, p = 1.0).
- A frozen backbone with a flow-matched residual, giving two outputs from one
  integration: a four-sample mean for viewing, within 0.25 dB of the best published
  network on lung PSNR, and a single sample for reporting, 1.5 to 2.2 times more
  accurate densitometrically.

## Layout

| path | what |
|---|---|
| `scte_r/` | model, losses, metrics, certificate (`agentic.py`), flow decoder |
| `scripts/` | data prep, operator fitting, certification, inference, figures |
| `baselines/` | vendored published networks (see `baselines/README.md`) |
| `上机手册.md` | **the one file to take on site — install, self-test, run, what to collect, troubleshooting, CLAIM checklist, reader-study protocol** |
| `PRIVATE_RUNBOOK.md` | background and rationale (not needed on site) |
| `STAGE1_执行报告.md` | full experimental record, including the negative results |

## Setup

```bash
pip install -r requirements.txt
bash scripts/fetch_baselines.sh      # I3Net is fetched, not redistributed
python scripts/preflight.py          # proves the pipeline works before you rely on it
```

`preflight.py` builds a synthetic pair with a known sub-slab z offset, runs the real
stage-1 code on it and checks the offset is recovered exactly, then fits the operator
and runs the certificate. See [上机手册.md](上机手册.md).

Going to a machine with no internet: `bash scripts/make_bundle.sh` builds a
self-contained tarball (code, weights, TotalSegmentator weights, wheels).

Everything needed on site is in a single document: [上机手册.md](上机手册.md).

## Data

No patient data is in this repository and none may be added — `.gitignore` blocks
`PRIVATE/`, every `*_thin/_thick/_lung/_base.npy`, and `ID_MAP_DO_NOT_EXPORT.csv`.
Development used the public RPLHR-CT dataset.
