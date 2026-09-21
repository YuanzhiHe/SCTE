% Generative thin-slice restoration recovers emphysema densitometry from thick-slice chest CT

## Abstract

Emphysema is quantified on chest computed tomography (CT) from the attenuation histogram of the lung, most often as the low-attenuation area below −950 Hounsfield units (HU), abbreviated LAA-950. The measurement requires 1 mm reconstructions, yet much of chest CT is reconstructed at 5 mm and hospitals without thin-slice capability cannot produce the thin series at all. Deep learning restores thin slices from thick ones and improves image quality, but whether that improvement reaches the measurement has not been tested. In an external cohort of 444 examinations from two manufacturers and two reconstruction kernels, it did not. A published restoration network exceeded Lanczos interpolation by 2.0 dB in peak signal-to-noise ratio (PSNR) while their LAA-950 biases differed by 0.07 percentage points, and each recovered about one-tenth of the bias carried by the unreconstructed 5 mm series. The cause lies in the structure of the problem itself. A deterministic estimator narrows the lung attenuation histogram and displaces it, and we measured that displacement at −16.32 ± 0.65 HU across the cohort, a spread too small for a patient-level effect. Threshold indices are the quantities most sensitive to both. Sampling the reconstruction residual with a flow-matching decoder removes the displacement and recovers 95% of the bias, at a PSNR below that of the unreconstructed series, while indicating scanner-level competence without a thin-slice reference.

## Introduction

The severity and distribution of pulmonary emphysema are read from the attenuation histogram of the lung on chest computed tomography (CT). Two families of index are in routine use. Threshold indices count the fraction of lung voxels below a fixed attenuation, in Hounsfield units (HU), with the low-attenuation area below −950 HU (LAA-950) the most widely reported and the low-attenuation area below −910 HU (LAA-910) used as a marker of air trapping. Percentile indices instead report the attenuation below which a fixed fraction of lung voxels falls, most often the 15th percentile (Perc15) and, less commonly, the 10th (Perc10). The Fleischner Society statement on CT-definable subtypes of chronic obstructive pulmonary disease (COPD) builds its visual and quantitative definitions on these indices^1^.

Both families are computed on the voxel histogram, so both depend on the size of the voxel. A 5 mm slice averages the attenuation of the tissue it contains, and low-attenuation voxels averaged with adjacent parenchyma move above the threshold. Gierada and colleagues measured this directly and showed that section thickness and reconstruction kernel shift the CT emphysema index by an amount that depends on the magnitude of the index itself^2^. Quantitative work therefore specifies 1 mm reconstructions.

Thick-slice CT nonetheless remains common in practice. Storage, dose, and reading time all favour it, and archives hold large volumes of 5 mm data acquired before quantitative analysis was contemplated. Hospitals without thin-slice reconstruction cannot generate the thin series at all, so for those sites the question is not whether to re-read an archive but whether any quantitative reading is possible.

Deep learning offers a route around the acquisition constraint. Networks that synthesize 1 mm CT from 5 mm CT now reach image quality close to the real thin series, and reader studies show that radiologists diagnose community-acquired pneumonia and detect nodules on synthesized thin slices about as well as on real ones^3^. A recent multicohort benchmark released a real-paired dataset for the task and reported that synthesized thin slices give segmentation-derived measurements and radiomic features closer to those from real 1 mm CT than interpolation does^4^. On the evidence available, restoration helps both readers and downstream quantitative pipelines.

Densitometry has not been part of that evidence. It is also the endpoint most exposed to one particular kind of error. Because it is a thresholded functional of the histogram rather than a shape or texture descriptor, a uniform shift of the histogram moves it while leaving every structural metric unchanged. Whether restoration networks improve densitometric agreement, leave it unchanged, or make it worse is therefore an open question, and the answer does not follow from their image quality.

We address it in an external cohort of 444 chest CT examinations from a hospital that contributed no training data, acquired on scanners from two manufacturers with two reconstruction kernels. This work makes three contributions. The first is an evaluation result: across two independent cohorts, the image quality gains of published restoration methods do not reach densitometric agreement, and a network with 2.0 dB more peak signal-to-noise ratio than interpolation carries the same LAA-950 bias to within 0.07 percentage points. The second is the account of why: a deterministic estimator narrows the lung attenuation histogram and displaces it by a constant, and threshold indices are the functionals most sensitive to both. We measure that displacement at −16.32 ± 0.65 HU over 444 examinations. The third is a method that acts on this account. Sampling the reconstruction residual with a flow-matching decoder recovers 95% of the bias carried by the unreconstructed 5 mm series, and the reference-free indicator that accompanies it tells a site whether the model transfers to its scanner before any thin-slice reference exists.

## Results

![**Fig. 1 | Study overview.** **a**, Cohorts. Hospital A contributed 203 paired examinations for model training; the public RPLHR-CT release contributed 50 whole volumes, which also supplied the pretraining weights used by the zero-shot arm. Hospital B, which contributed no training data, provided 494 paired examinations on scanners from two manufacturers with two reconstruction kernels; 50 were used to fit protocol constants and the indicator thresholds and 444 were reported, of which 311 were soft-kernel and 126 sharp-kernel. Descriptive statistics use all 444 examinations and paired tests use the 438 that every arm completed with finite values throughout. In every cohort the thick and thin series of an examination come from the same raw acquisition. **b**, Evaluation framework. The 5 mm series as acquired is the input to every arm, the matched 1 mm series is the reference, and each arm is assessed on the same set of endpoints. TVSRN and I3Net were evaluated on the public cohort, where released weights are available.](figures/fig1_overview.pdf)

### Study cohorts

The validation cohort comprised 444 paired chest CT examinations from a hospital that contributed no training data, acquired on scanners from two manufacturers with two reconstruction kernels (Fig. 1a). A further 50 examinations from the same hospital were used to fit protocol constants and indicator thresholds and are not included in any reported statistic. A public paired cohort of 50 whole volumes supported the comparison with methods whose weights have been released. Every arm took the 5 mm series as acquired and was assessed against the matched 1 mm series on the same endpoints (Fig. 1b). In the validation cohort the 1 mm reference gave LAA-950 7.82 ± 8.21%, LAA-910 19.72 ± 14.42%, and Perc15 −911.2 ± 43.3 HU.

![**Fig. 2 | Densitometric agreement in the external cohort.** **a**, Coronal views of one examination from the public paired cohort, the plane in which 5 mm and 1 mm acquisition differ. Columns are the 5 mm series upsampled, CTHNet, SCTE-R and the 1 mm reference; the lower row magnifies the boxed parenchymal region. **b**, Fraction of lung voxels below a given attenuation for the same examination, on a logarithmic vertical axis, with the two standard cut-offs marked. **c**, Peak signal-to-noise ratio against absolute LAA-950 bias in the external cohort ($n$ = 444); the dashed line marks the mean bias of the three non-generative arms and the curved arrow the distance covered by SCTE-R from the unreconstructed series. Circles are generative arms, squares the others. **d**, Bias and 95% limits of agreement for LAA-950 ($n$ = 444); points are the bias, bars the limits, and the dashed line zero bias. **e**, Global attenuation displacement measured directly ($n$ = 444); bars are means and error bars standard deviations. **f**, Fraction of examinations passing the competence indicator in four model-by-domain combinations, with hatching on the domain in which the densitometric endpoints failed. Panels **a** and **b** come from the public paired cohort and panels **c** to **f** from the external cohort; these are not the same participants.](figures/fig2_results.pdf)

### Reading the 5 mm series directly

Through-plane views make the difference visible before any index is computed: the 5 mm series carries a staircase artefact that the reconstructions do not, and the parenchymal texture of the sampler resembles the reference while that of the deterministic backbone is smoothed (Fig. 2a). Measured on the acquired 5 mm series with no reconstruction, LAA-950 read 2.83% against a reference of 7.82%, a bias of −4.99 percentage points with 95% limits of agreement (LoA) of −13.50 to +3.51 and a concordance correlation coefficient (CCC) of 0.631 (Table 1). About two-thirds of the emphysema present was therefore not counted. Perc15 was overestimated by 25.7 HU and Perc10 by 29.9 HU (Table 3). Under the stricter rule that all five sub-voxels of a thick voxel be lung, the mean absolute error moved from 5.05 to 5.02 percentage points. The bias therefore lies in through-plane partial-volume averaging, not in the rule used to define the lung set.

### Image quality gains do not reach the measurement

Relative to reading the 5 mm series directly, Lanczos interpolation recovered 12% of the LAA-950 bias and a published restoration network recovered 10%, while the network exceeded interpolation by 2.0 dB in PSNR (Fig. 2c, Table 1). The two differed in LAA-950 bias by 0.07 percentage points. Across the four arms the relation between PSNR and densitometric bias was flat over a 3.2 dB span.

The public cohort showed the same ordering with far smaller absolute differences (Fig. 3a). Three published methods reached PSNR above 33 dB, about 3 dB above Lanczos, and all three had slightly larger LAA-950 bias than Lanczos (−0.82, −0.90 and −0.86 against −0.78 percentage points; Table 4). That cohort has a reference LAA-950 of 1.21 ± 1.14%, close to a normal population, so every method's error is small in absolute terms and the failure is invisible there. The difference in scale between the two cohorts is itself informative about why the gap has not been reported.

### A constant attenuation displacement

On the validation cohort the deterministic regression displaced the lung histogram by −16.32 ± 0.65 HU across 444 examinations (Fig. 2e, Table 5). A standard deviation of 0.65 HU on a −16 HU effect identifies the displacement as a constant of the estimator, with case-level variation an order of magnitude below it. On the public cohort, adding the densitometric indices to the training objective and varying their weight left the displacement within 0.01 HU of its original value, so the effect is not an artefact of the loss used. A control that applied a fitted scalar offset and nothing else displaced the histogram by −39.8 to −41.8 HU, confirming that a scalar correction alone is not a solution.

### Restored agreement

The residual sampler, applied zero-shot with weights trained only on public data, gave an LAA-950 bias of −0.26 percentage points with LoA of −5.61 to +5.08 and CCC 0.933, against −4.48 and 0.682 for the published network (Table 1, Fig. 2d). LAA-910 bias was −0.32 percentage points with CCC 0.986 (Table 2, Fig. 3b), and Perc15 bias was +2.5 HU against +19.3 HU (Table 3, Fig. 3c). Sweeping the threshold instead of fixing it gave the same separation across the whole low-attenuation region, with the 5 mm and backbone curves below the reference throughout and the sampler following it (Fig. 2b).

![**Fig. 3 | Endpoints not covered by Fig. 2.** **a**, Image quality against absolute LAA-950 bias on the public paired cohort ($n$ = 50). The three published methods take almost the same value and carry one shared label. **b**, Bias and 95% limits of agreement for LAA-910 in the external cohort ($n$ = 444). **c**, Bias in the 15th and 10th percentiles of the lung attenuation histogram in the external cohort ($n$ = 444); columns left to right are the 5 mm series read directly, Lanczos interpolation, CTHNet and SCTE-R applied zero-shot.](figures/fig3_additional.pdf) Expressed against the unreconstructed series, the sampler recovered 95% of the LAA-950 bias. Its PSNR was 26.81 dB, below the 27.59 dB of the unreconstructed 5 mm series and 3.96 dB below the published network.

Agreement also held at the level of the individual lobe. Across the five lobes CCC ranged from 0.929 to 0.956 with biases between +0.84 and +1.50 percentage points (Table 6). Results were insensitive to the number of integration steps, with a paired absolute-error difference in LAA-950 between 32 and 64 integration steps of −0.06 to +0.46 percentage points (bootstrap 95% confidence interval, 20 examinations) and a lung PSNR difference of 0.09 dB.

Fine-tuning on paired data from the validation site tightened the LAA-950 limits of agreement to −3.18 to +5.26 and raised CCC to 0.948, and raised PSNR by 1.5 dB, at the cost of larger LAA-910 bias and larger percentile errors. Sites that can produce paired data can therefore trade one endpoint against another; sites without thin-slice capability cannot produce such data at all, which is why the zero-shot arm is the deployable one.

### Competence varies by scanner, not by scan

The indicator passed 80% of examinations in the domain the public model was trained on. On the validation cohort it passed 16.1% of the 311 soft-kernel examinations, where the densitometric endpoints held with a diagnostic reclassification rate of 1.6%, and 4.8% of the 126 sharp-kernel examinations, where they did not, with LAA-950 bias between −4 and −5.5 percentage points and a reclassification rate of 27.6% (Fig. 2f). After fine-tuning on matched data both scanner families passed at about 99% with endpoints intact. The sharp-kernel data lay inside the physical calibration set, so what the indicator separates is whether the model has seen this kind of data, not whether the protocol has seen this scanner.

Within a single scanner the verdict carried no case-level information. Among the 311 soft-kernel examinations, 0 of the 50 that passed and 5 of the 261 that were flagged crossed the 10% diagnostic cut-off (Fisher exact test, two-sided, p = 1.0). The indicator therefore functions as a site-level gate.

Table 1 | Agreement with the 1 mm reference for LAA-950, external cohort (n = 444). Bias is reconstructed minus reference in percentage points (pp); reference LAA-950 was 7.82 ± 8.21%.

| Arm | Bias (pp) | 95% LoA (pp) | CCC |
|:---|---:|:---:|---:|
| 5 mm read directly | −4.99 | −13.50 to +3.51 | 0.631 |
| Lanczos interpolation | −4.41 | −12.32 to +3.50 | 0.689 |
| CTHNet | −4.48 | −12.80 to +3.84 | 0.682 |
| SCTE-R, zero-shot | −0.26 | −5.61 to +5.08 | 0.933 |
| SCTE-R, site-adapted | +1.04 | −3.18 to +5.26 | 0.948 |

Table 2 | Agreement with the 1 mm reference for LAA-910, external cohort (n = 444). Reference LAA-910 was 19.72 ± 14.42%.

| Arm | Bias (pp) | 95% LoA (pp) | CCC |
|:---|---:|:---:|---:|
| 5 mm read directly | −8.37 | −16.49 to −0.26 | 0.801 |
| Lanczos interpolation | −7.30 | −14.59 to −0.01 | 0.841 |
| CTHNet | −5.98 | −13.12 to +1.17 | 0.889 |
| SCTE-R, zero-shot | −0.32 | −4.85 to +4.22 | 0.986 |
| SCTE-R, site-adapted | +1.72 | −3.08 to +6.52 | 0.978 |

Table 3 | Bias in the low-attenuation percentiles, external cohort (n = 444). Reference Perc15 was −911.2 ± 43.3 HU.

| Arm | Perc15 bias (HU) | Perc10 bias (HU) |
|:---|---:|---:|
| 5 mm read directly | +25.7 | +29.9 |
| Lanczos interpolation | +23.5 | +26.8 |
| CTHNet | +19.3 | +23.8 |
| SCTE-R, zero-shot | +2.5 | +2.0 |

Table 4 | Public paired cohort (n = 50), where the reference LAA-950 was 1.21 ± 1.14%. All methods used the authors' released implementations, retrained and evaluated on one split. CCC carries between-subject variance, so the values here share a denominator with each other but not with Tables 1 and 2.

| Method | PSNR (dB) | Bias (pp) | Mean absolute error (pp) | CCC |
|:---|---:|---:|---:|---:|
| CTHNet | 33.44 | −0.82 | 0.82 | 0.152 |
| TVSRN | 33.17 | −0.90 | 0.90 | 0.068 |
| I3Net | 33.22 | −0.86 | 0.86 | 0.100 |
| Lanczos interpolation | 30.27 | −0.78 | 0.78 | 0.259 |
| SCTE-R | 30.83 | −0.51 | 0.54 | 0.594 |

Table 5 | Global attenuation displacement measured directly, external cohort (n = 444). Values are mean ± standard deviation across examinations.

| Arm | Displacement δ (HU) |
|:---|---:|
| Deterministic regression | −16.32 ± 0.65 |
| Scalar-offset control | −39.8 to −41.8 |
| SCTE-R | −1.76 ± 1.06 |

Table 6 | Lobar agreement for LAA-950, site-adapted arm, external cohort (n = 444).

| Lobe | Bias (pp) | CCC |
|:---|---:|---:|
| Left upper | +0.84 | 0.951 |
| Left lower | +1.33 | 0.929 |
| Right upper | +0.85 | 0.949 |
| Right middle | +0.85 | 0.956 |
| Right lower | +1.50 | 0.954 |

## Discussion

Image quality and densitometric accuracy came apart on this task, and by a wide margin. A published restoration network with 2.0 dB more PSNR than Lanczos interpolation carried the same LAA-950 bias to within 0.07 percentage points, and the method that restored agreement had the lowest PSNR of every arm tested, including the unreconstructed 5 mm series. Anyone selecting a restoration model on image quality alone would rank these methods in close to the reverse of their densitometric order.

Two properties of a deterministic estimator account for this. The first follows from the objective: minimising squared error suppresses the residual component the operator does not constrain, which narrows the lung histogram and leaves less mass below the threshold. The second property is an empirical regularity. Across 444 examinations the estimator displaced the histogram by −16.32 HU with a standard deviation of 0.65 HU, a spread far too small for a patient-level effect, and adding the densitometric indices to the training objective left that displacement within 0.01 HU. Threshold statistics are the functionals such a displacement moves, which is why they fail while structural metrics do not. A sampled residual retains its width, and under sampling the displacement fell to −1.76 HU.

The magnitude of the displacement remains an open question. It was similar in the public cohort, where the model was in domain, which leaves domain shift as at most a partial account.

This scopes rather than contradicts the benchmark result that restoration improves downstream consistency^4^. Segmentation-derived measurements and radiomic features are shape and texture descriptors, largely invariant to a uniform attenuation shift, and there is no reason for them to degrade. Threshold densitometry is the endpoint that a shift moves by construction. Downstream endpoints are not interchangeable as validation, and the ones that stress a restoration hardest are those computed from the voxel histogram.

For sites without thin-slice capability, the practical reading is that reconstruction is worth doing and the choice of method matters more than its image quality suggests. Reading the 5 mm series directly missed about two-thirds of the emphysema present. Interpolation and a published network each recovered about a tenth of that bias; the residual sampler recovered 95%. The accompanying indicator addresses the question such a site must answer first, which is whether the model transfers to its scanner at all, and it answers that question without a thin-slice reference. Its resolution is the scanner, which is the granularity at which a site makes the deployment decision.

Deep-learning restoration of thin-slice chest CT improves image quality without improving emphysema densitometry. A deterministic estimator narrows the lung histogram and, as measured here, displaces it by a constant amount; threshold indices are the quantities most sensitive to both. Sampling the reconstruction residual removes the displacement and recovers 95% of the bias present in the unreconstructed 5 mm series, on an external cohort of 444 examinations from two manufacturers, at an image quality below that of the series it started from. A reference-free indicator identifies scanners on which the model is not competent.

Two directions follow from this work. Multi-site evaluation would map the range of protocols over which the fitted operator and the indicator thresholds carry, beyond the two manufacturers and two kernels covered here, and the indicator itself offers a way to screen each new site before any thin-slice reference exists. A reading study, together with comparison against pulmonary function or longitudinal decline, would then establish that the recovered agreement changes what a clinician concludes.

## Methods

![**Fig. 4 | Method.** **a**, Forward model. Thick-slice reconstruction applies a through-plane sensitivity profile of full width at half maximum $w$ and then averages each group of $r = 5$ slices, giving $y = A_w x + \varepsilon$. The operator preserves slab means and has a non-trivial null space, so through-plane detail that averages to zero within a slab leaves no trace in the observation. **b**, Residual-space sampling, shown unrolled. A base predictor supplies $x_0$; the scaled residual is initialised from a standard normal and advanced by explicit Euler steps of a velocity field $v_\theta$ conditioned on the observation and the fitted width, with a projection onto the measurement-consistent set after each step. The estimate is $\hat{x} = x_0 + s_r u$, and sampling rather than averaging retains the width of the residual distribution. **c**, The three reference-free terms are parallel readouts of the same data-consistency residual $R = A_w\hat{x} - y$ taken over the eroded lung interior, and the verdict requires all three to fall inside thresholds fitted on the null distribution of calibration cases.](figures/fig4_method.pdf)

### Related work

Restoration of 1 mm CT from thicker reconstructions is usually posed as through-plane super-resolution. The RPLHR-CT dataset introduced a real-paired benchmark, in which the thick and thin series come from the same raw acquisition rather than from simulated downsampling, together with a transformer baseline^5^. Subsequent architectures include an inter-intra-slice interpolation network^6^, a three-dimensional conditional generative adversarial network applied to spinal morphology^7^, and a convolutional-transformer hybrid trained to recover masked regions from visible ones, which reached reader-level parity with real thin slices for pneumonia diagnosis and nodule detection^3^. A multicohort benchmark later released a multi-ratio real-paired dataset and showed that training on simulated rather than real thick-slice inputs degrades every method tested, which establishes real pairs as the appropriate training and evaluation setting^4^.

Evaluation across this literature rests on peak signal-to-noise ratio (PSNR) and the structural similarity index measure (SSIM), supplemented by task-level endpoints. Synthesized thin slices raised nodule-level sensitivity of a computer-aided detection system on 5 mm chest CT from 0.826 to 0.916 at two false positives per scan^8^, and improved the reproducibility and discriminative performance of radiomic features in nodule assessment^9^. None of these endpoints is a thresholded statistic of the attenuation histogram.

That thickness perturbs emphysema indices is established. The effect is driven by partial-volume averaging within the voxel and scales with the index^2^. Related work has addressed the neighbouring problem of reconstruction kernel, using image-to-image translation to convert archived CT to a kernel for which quantitative reference values exist^10^. A systematic review of deep-learning image reconstruction for dose reduction found that it alters abdominal CT densitometry as well as image quality^11^. In both settings the perturbation is tied to a reconstruction setting that can be held fixed or converted. Slice thickness differs in that the averaging acts on the lung parenchyma itself, so there is no unaffected region to calibrate against.

Generative reconstruction can synthesize structures that are absent from the measurement. The decomposition of an estimate into measurement-space and null-space components shows that such fabrications live in the null space of the forward operator, where data consistency is blind to them^12^. This result bounds what any consistency-based check can certify, and it sets the scope of the indicator introduced below.

### Cohorts and endpoints

The training cohort comprised 203 paired examinations from one hospital. The validation cohort comprised 494 paired examinations from a second, independent hospital, of which 50 were used to fit protocol constants and indicator thresholds and 444 were reported. Validation examinations came from two manufacturers and two reconstruction kernels. A public paired cohort of 50 whole volumes was used for the comparison with methods whose published weights were available. In every cohort the thick and thin series of an examination were reconstructed from the same raw acquisition, so the two series are registered by construction and no inter-scan alignment was required. Every index was computed on whole volumes over lung lobe masks from TotalSegmentator^13^, not over an attenuation window. In the validation cohort the 1 mm reference gave LAA-950 7.82 ± 8.21%, LAA-910 19.72 ± 14.42%, and Perc15 −911.2 ± 43.3 HU.

### Problem formulation

Let $x \in \mathbb{R}^{D \times H \times W}$ denote the 1 mm volume and $y \in \mathbb{R}^{(D/r) \times H \times W}$ the 5 mm volume reconstructed from the same acquisition, where $r = 5$ is the ratio of slice spacings and $D$, $H$, $W$ are the through-plane and in-plane dimensions. Thick-slice reconstruction applies a through-plane sensitivity profile followed by decimation (Fig. 4a). We model it as

$$y = A_w x + \varepsilon,$$

where $A_w$ convolves along the through-plane axis with a Gaussian of full width at half maximum $w$ and then averages each group of $r$ slices, and $\varepsilon$ collects acquisition noise. The effective width $w$ is a protocol constant, fitted once per protocol on calibration pairs.

Two properties of $A_w$ carry the argument that follows. It is mean-preserving, so the mean attenuation of any slab is unchanged by the operator. And it has a non-trivial null space: through-plane detail that averages to zero within a slab leaves no trace in $y$.

The quantity to be recovered is not $x$ but a functional of it. For a lung set $\Omega$ and threshold $\tau$,

$$\mathrm{LAA}_\tau(x) = \frac{|\{v \in \Omega : x_v < \tau\}|}{|\Omega|},$$

and Perc15 is the corresponding quantile. Both depend on $x$ only through the histogram of $\{x_v : v \in \Omega\}$, and both respond only to histogram mass that crosses $\tau$; mass that moves elsewhere leaves them unchanged.

This is where a squared-error estimator and a densitometric estimator part company. Write the reconstruction as $\hat{x} = x + e$. Mean squared error is minimised by the conditional mean, which suppresses the component of $e$ that the operator does not constrain. That suppression narrows the lung histogram, and a narrower histogram places less mass below $\tau$, so a reconstruction that is optimal in squared error underestimates LAA by construction.

A second effect is an empirical regularity. Deterministic estimators on this task also settle at a non-zero global displacement

$$\delta = \mathbb{E}_\Omega[\hat{x} - x],$$

which shifts every threshold index at once and to which threshold indices are the most sensitive functionals. We measure $\delta$ below and report how far it varies between examinations and whether the training objective moves it.

Sampling addresses both effects at once, because a draw from the posterior retains the width of the residual distribution instead of collapsing it, and therefore has nothing to gain from a displacement.

### Restoration by residual sampling

The pipeline is summarised in Fig. 4b. Rather than regress $x$, we sample it. A base predictor supplies $x_0$, either through-plane upsampling of $y$ or a frozen restoration backbone. We work in the scaled residual

$$u = \frac{x - x_0}{s_r},$$

with $s_r$ a fixed scale, and learn a velocity field $v_\theta(u, t \mid y, w)$ with parameters $\theta$ by flow matching^14^, integrating over $t \in [0, 1]$ with an explicit Euler scheme. The scale $s_r$ is fixed and is distinct from the tail term $s$ introduced below.

After each integration step the iterate is projected back onto the set consistent with the measurement, which keeps $A_w \hat{x}$ close to $y$ while leaving the null-space component free for the sampler to populate.

### Reference-free competence indicator

Mean preservation gives an estimator of $\delta$ that needs no 1 mm reference (Fig. 4c). Because $A_w$ preserves slab means, any displacement of $\hat{x}$ appears undiminished in $A_w\hat{x} - y$ over the eroded lung set $\Omega_e$, so

$$\hat{\delta} = \frac{1}{|\Omega_e|}\sum_{v \in \Omega_e} \left(A_w\hat{x} - y\right)_v$$

is computable at deployment. Two further quantities accompany the displacement estimate. The structural residual $\rho_{\text{struct}} = \lVert (A_w\hat{x} - y) - \hat{\delta}\rVert / \lVert y - \bar{y}\rVert$ measures what remains after the displacement is removed, and $s$ compares the through-plane variance of $\hat{x}$ with the value predicted from $y$ by a calibration fitted on paired data, expressed in units of that calibration's held-out residual spread. Thresholds for all three come from the null distribution obtained by passing the true thin volume through the same estimators on calibration cases.

Two limits follow from the null-space argument^12^. All three quantities are computed from $A_w\hat{x} - y$ or from a variance predicted by $y$, so none of them sees the null space directly. The indicator therefore operates at the level of the scanner, and that is the level at which we report it.

### Statistical analysis

The independent unit is the examination. Agreement with the 1 mm reference is reported as bias, defined as the mean of reconstructed minus reference across examinations, as 95% limits of agreement, defined as bias ± 1.96 standard deviations of the per-examination difference in the sense of Bland and Altman^15^, and as the concordance correlation coefficient of Lin^16^. Descriptive statistics for each arm are computed on all examinations that arm completed, which is 444 in every case.

Paired comparisons between arms use the 438 examinations that every arm completed with finite values throughout. Six examinations were excluded under a rule fixed before the comparison: an examination is dropped from every endpoint as soon as any arm returns a non-finite value for it anywhere, because LAA treats a non-finite voxel as parenchyma above the threshold and would otherwise report a finite but wrong value. Four of the six arose from non-finite voxels in a reconstruction and two from a lobe too small to support a stable index. The reference distribution of the 438-examination subset, LAA-950 7.82 ± 8.14%, matches that of the full cohort. Differences in absolute error between arms are summarised by the bootstrap percentile 95% confidence interval over 2000 resamples of examinations with a fixed seed. Comparisons of proportions use the Fisher exact test, two-sided.

We report effect estimates with intervals rather than significance alone, and no multiplicity correction is applied because no claim in this work rests on a threshold crossed by a single p value; the agreement statistics that carry the findings are point estimates with intervals across the full cohort. Analyses were performed in Python 3.11 with NumPy and SciPy.

## Data availability

The public paired cohort used for the cross-method comparison is available from the RPLHR-CT release^5^. The two clinical cohorts comprise identifiable chest CT held under institutional agreements that reserve the imaging to the contributing hospitals, and they are not available for release. Derived per-examination measurements underlying every reported statistic may be requested from the corresponding author and will be shared subject to approval by the contributing institutions.

## Code availability

Training, inference, calibration and evaluation code, together with the model weights used for the zero-shot arm, are available at https://github.com/YuanzhiHe/SCTE. Experiments were implemented in Python 3.11 with PyTorch and run on a single NVIDIA GPU.

## References

1. Lynch, D. A. et al. CT-definable subtypes of chronic obstructive pulmonary disease: a statement of the Fleischner Society. *Radiology* **277**, 192–205 (2015).
2. Gierada, D. S. et al. Effects of CT section thickness and reconstruction kernel on emphysema quantification: relationship to the magnitude of the CT emphysema index. *Acad. Radiol.* **17**, 146–156 (2010).
3. Yu, P. et al. Spatial resolution enhancement using deep learning improves chest disease diagnosis based on thick slice CT. *npj Digit. Med.* **7**, 335 (2024).
4. Yu, P. et al. Benchmarking AI-generated thin-slice CT under clinical reconstruction conditions: a multicohort study. *npj Digit. Med.* (2026).
5. Yu, P. et al. RPLHR-CT dataset and transformer baseline for volumetric super-resolution from CT scans. In *Medical Image Computing and Computer Assisted Intervention* 344–353 (2022).
6. Song, H., Mao, X., Yu, J., Li, Q. & Wang, Y. I3Net: inter-intra-slice interpolation network for medical slice synthesis. *IEEE Trans. Med. Imaging* **43**, 3306–3318 (2024).
7. Nakamoto, A. et al. Three-dimensional conditional generative adversarial network-based virtual thin-slice technique for the morphological evaluation of the spine. *Sci. Rep.* **12**, 12176 (2022).
8. Jeong, J. et al. Deep learning-based slice thickness reduction for computer-aided detection of lung nodules in thick-slice CT. *Diagnostics* **14**, 2558 (2024).
9. Yang, H. et al. Deep learning-based CT slice synthesis improves radiomic feature reproducibility and discriminative performance in lung nodule assessment. *Insights Imaging* **17** (2026).
10. Tanabe, N. et al. Kernel conversion for robust quantitative measurements of archived chest computed tomography using deep learning-based image-to-image translation. *Front. Artif. Intell.* **4**, 769557 (2022).
11. van Stiphout, J. A. et al. The effect of deep learning reconstruction on abdominal CT densitometry and image quality: a systematic review and meta-analysis. *Eur. Radiol.* **32**, 2921–2929 (2021).
12. Bhadra, S., Kelkar, V. A., Brooks, F. J. & Anastasio, M. A. On hallucinations in tomographic image reconstruction. *IEEE Trans. Med. Imaging* **40**, 3249–3260 (2021).
13. Wasserthal, J. et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. *Radiol. Artif. Intell.* **5**, e230024 (2023).
14. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow matching for generative modeling. In *International Conference on Learning Representations* (2023).
15. Bland, J. M. & Altman, D. G. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet* **327**, 307–310 (1986).
16. Lin, L. I.-K. A concordance correlation coefficient to evaluate reproducibility. *Biometrics* **45**, 255–268 (1989).

## Acknowledgements

## Author contributions

## Competing interests

The authors declare no competing interests.
