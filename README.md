# SCTE-R — thick-slice CT restoration for emphysema densitometry

Restoring 1 mm CT from 5 mm reconstructions so that the *emphysema indices*
(LAA-950, LAA-910, Perc15/10/5) are right — not just the pixels.

## The finding this is built on

Three published thick-to-thin networks, retrained on one split and scored under one
protocol (50 whole volumes, TotalSegmentator lobe masks):

| method | lung PSNR | LAA-950 MAE | **LAA-950 CCC** |
|---|---|---|---|
| Lanczos-3 interpolation | 27.65 | 0.78 | 0.259 |
| TVSRN (MICCAI 2022) | 31.08 | 0.90 | 0.068 |
| CTHNet (npj Digit. Med. 2024) | 31.60 | 0.82 | 0.152 |
| I3Net (IEEE TMI 2024) | 31.34 | 0.86 | 0.100 |

Each beats interpolation by 3.4–4.0 dB and **none improves the densitometry** —
their LAA-950 concordance is *below* plain interpolation. Image fidelity and
density fidelity are not the same objective.

Why: the map from reconstruction to a threshold statistic is many-to-one, and the
cheapest way to hit the target is a global HU displacement. Every deterministic
regression we trained lands on a ~20 HU displaced solution regardless of the loss
(halving the biomarker loss weight moves it from 19.08 to 19.09 HU).

## What this repo adds

- **A per-scan, reference-free certificate.** The displacement `δ̂` is measurable
  from the thick observation alone, via the forward operator's mean preservation.
  It separates "recovered the distribution" from "shifted the distribution":
  displaced arms 0/50 certified, ours 47/50, with `|δ̂|` inside the
  perfect-reconstruction null.
- **A strong backbone plus a flow-matched residual**, emitting two outputs from one
  integration: a 4-sample mean for viewing (lung PSNR within 0.25 dB of the best
  published network) and a single sample for reporting (densitometry 1.5–2.2×
  more accurate, CCC 3.9×).

## Layout

| path | what |
|---|---|
| `scte_r/` | model, losses, metrics, certificate (`agentic.py`), flow decoder |
| `scripts/` | data prep, operator fitting, certification, inference, figures |
| `baselines/` | vendored published networks (see `baselines/README.md`) |
| `DEPLOY.md` | **deployment and on-site execution — start here** |
| `PRIVATE_RUNBOOK.md` | background and rationale for the on-site procedure |
| `STAGE1_执行报告.md` | full experimental record, including the negative results |

## Setup

```bash
pip install -r requirements.txt
bash scripts/fetch_baselines.sh      # I3Net is fetched, not redistributed
python scripts/preflight.py          # proves the pipeline works before you rely on it
```

`preflight.py` builds a synthetic pair with a known sub-slab z offset, runs the real
stage-1 code on it and checks the offset is recovered exactly, then fits the operator
and runs the certificate. See [DEPLOY.md](DEPLOY.md).

Going to a machine with no internet: `bash scripts/make_bundle.sh` builds a
self-contained tarball (code, weights, TotalSegmentator weights, wheels).

## Data

No patient data is in this repository and none may be added — `.gitignore` blocks
`PRIVATE/`, every `*_thin/_thick/_lung/_base.npy`, and `ID_MAP_DO_NOT_EXPORT.csv`.
Development used the public RPLHR-CT dataset.
