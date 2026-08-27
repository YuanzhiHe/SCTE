# Vendored published baselines

Unmodified model definitions from the authors' official releases, kept here so the
comparison is reproducible offline. Only the surrounding data/training code is ours;
the networks themselves are byte-identical to the upstream files.

| dir | method | paper | upstream |
|---|---|---|---|
| `tvsrn/` | TVSRN | Yu et al., *RPLHR-CT Dataset and Transformer Baseline for Volumetric Super-Resolution from CT Scans*, MICCAI 2022 | github.com/smilenaxx/RPLHR-CT |
| `cthnet/` | CTHNet (`TVSRN3d_v1`) | Yu et al., *Spatial resolution enhancement using deep learning improves chest disease diagnosis based on thick slice CT*, npj Digital Medicine 7:335 (2024) | github.com/smilenaxx/CTHNet-for-CT-Slice-Thickness-Reduction |

Each directory keeps the upstream LICENSE.

**Alignment convention (important).** Both repos index a 5 mm slice `k` against thin
slice `5k`, and supervise the output on thin slices `5k+3 … 5k+12` given thick slices
`k … k+3` (see `get_train_img` upstream: `mask_z_s = z_s*5+3`, `mask_z_e = (z_e-1)*5-2`).
Our adapter reproduces that indexing exactly. Using our own centre convention
(`5k+2`) instead would shift the target by two slices and silently handicap them.

| `i3net/` | I3Net | Song et al., *I3Net: Inter-Intra-slice Interpolation Network for Medical Slice Synthesis*, IEEE TMI 2024 | github.com/eeeric-code/I3Net |

**I3Net's built-in assumption.** Its forward ends with `out[:, ::upscale] = x`: the
low-resolution slices are *copied* into the output at stride `upscale`. That is
correct when the thick series is a SUBSAMPLING of the thin one, which is how the
method is trained in the paper (pseudo-LR). Our thick slices are real 5 mm
reconstructions - slab averages, not samples - so that hard copy pins 1/5 of the
output slices to a value they do not actually have. We evaluate the method exactly
as published and report this, rather than quietly deleting the line: the mismatch
between pseudo-LR and real thick slices is a property of the method, not a bug in
our harness.

## Licensing / redistribution

`tvsrn/` and `cthnet/` are Apache-2.0 and are vendored here with their upstream
LICENSE files, so the comparison reproduces offline.

`i3net/` is **not** redistributed: the upstream repository ships no LICENSE, which
by default reserves all rights. Fetch it once, on a machine with internet:

```bash
bash scripts/fetch_baselines.sh
```

Everything else in the pipeline works without it; only the I3Net baseline column
requires this step.
