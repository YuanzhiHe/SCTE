% Generative thin-slice restoration recovers emphysema densitometry from thick-slice chest CT

## Abstract

Emphysema is quantified on chest computed tomography (CT) from the attenuation histogram of the lung, most often as the low-attenuation area below −950 Hounsfield units (HU), abbreviated LAA-950. The measurement requires 1 mm reconstructions, yet much of chest CT is reconstructed at 5 mm and hospitals without thin-slice capability cannot produce the thin series at all. Deep learning restores thin slices from thick ones and improves image quality, but whether that improvement reaches the measurement has not been tested. In an external cohort of 444 examinations from two manufacturers and two reconstruction kernels, it did not. A published restoration network exceeded Lanczos interpolation by 2.0 dB in peak signal-to-noise ratio (PSNR) while their LAA-950 biases differed by 0.07 percentage points, and each recovered about one-tenth of the bias carried by the unreconstructed 5 mm series. The cause lies in the problem rather than the network. The optimum of a deterministic regression here is a constant displacement of the attenuation histogram, measured at −16.32 ± 0.65 HU, and threshold indices are the most sensitive quantities to it. Sampling the reconstruction residual with a flow-matching decoder removes the displacement and recovers 95% of the bias, at a PSNR below that of the unreconstructed series, while indicating scanner-level competence without a thin-slice reference.

## Introduction

The severity and distribution of pulmonary emphysema are read from the attenuation histogram of the lung on chest computed tomography (CT). Two families of index are in routine use. Threshold indices count the fraction of lung voxels below a fixed attenuation, in Hounsfield units (HU), with the low-attenuation area below −950 HU (LAA-950) the most widely reported and the low-attenuation area below −910 HU (LAA-910) used as a marker of air trapping. Percentile indices instead report the attenuation below which a fixed fraction of lung voxels falls, most often the 15th percentile (Perc15). The Fleischner Society statement on CT-definable subtypes of chronic obstructive pulmonary disease (COPD) builds its visual and quantitative definitions on these indices[^lynch2015].

Both families are computed on the voxel histogram, so both depend on the size of the voxel. A 5 mm slice averages the attenuation of the tissue it contains, and low-attenuation voxels averaged with adjacent parenchyma move above the threshold. Gierada and colleagues measured this directly and showed that section thickness and reconstruction kernel shift the CT emphysema index by an amount that depends on the magnitude of the index itself[^gierada2010]. Quantitative work therefore specifies 1 mm reconstructions.

Thick-slice CT nonetheless remains common. Storage, dose, and reading time all favour it, and archives hold large volumes of 5 mm data acquired before quantitative analysis was contemplated. Hospitals without thin-slice reconstruction cannot generate the thin series at all, so for those sites the question is not whether to re-read an archive but whether any quantitative reading is possible.

Deep learning offers a route. Networks that synthesize 1 mm CT from 5 mm CT now reach image quality close to the real thin series, and reader studies show that radiologists diagnose community-acquired pneumonia and detect nodules on synthesized thin slices about as well as on real ones[^yu2024cthnet]. A recent multicohort benchmark released a real-paired dataset for the task and reported that synthesized thin slices give segmentation-derived measurements and radiomic features closer to those from real 1 mm CT than interpolation does[^yu2026benchmark]. On the evidence available, restoration helps both readers and downstream quantitative pipelines.

Densitometry has not been part of that evidence. It is also the endpoint with the least room for error, because it is a thresholded functional of the histogram rather than a shape or a texture descriptor, and a uniform shift of the histogram moves it without changing anything a structural metric would detect. Whether restoration networks improve densitometric agreement, leave it unchanged, or make it worse is therefore an open question, and the answer does not follow from their image quality.

We address it in an external cohort of 444 chest CT examinations from a hospital that contributed no training data, acquired on scanners from two manufacturers with two reconstruction kernels. We report three findings. First, image quality gains from published restoration methods do not transfer to densitometric agreement. Second, the reason is a property of the inverse problem rather than of any particular network, and it is measurable. Third, a residual-space generative decoder removes the responsible error and restores agreement, while the accompanying reference-free indicator tells a site whether the model transfers to its scanner before any thin-slice reference exists.

## Related Works

### Thick-to-thin slice restoration

Restoration of 1 mm CT from thicker reconstructions is usually posed as through-plane super-resolution. The RPLHR-CT dataset introduced a real-paired benchmark, in which the thick and thin series come from the same raw acquisition rather than from simulated downsampling, together with a transformer baseline[^yu2022tvsrn]. Subsequent architectures include an inter-intra-slice interpolation network[^song2024i3net] and a convolutional-transformer hybrid trained to recover masked regions from visible ones, which reached reader-level parity with real thin slices for pneumonia diagnosis and nodule detection[^yu2024cthnet]. A multicohort benchmark later released a multi-ratio real-paired dataset and showed that training on simulated rather than real thick-slice inputs degrades every method tested, which establishes real pairs as the appropriate training and evaluation setting[^yu2026benchmark].

Evaluation across this literature rests on peak signal-to-noise ratio (PSNR) and the structural similarity index measure (SSIM), supplemented by reader studies and, more recently, by segmentation and radiomic consistency. None of these endpoints is a thresholded statistic of the attenuation histogram.

### Slice thickness and quantitative CT

That thickness perturbs emphysema indices is established. The effect is driven by partial-volume averaging within the voxel and scales with the index[^gierada2010]. Related work has addressed the neighbouring problem of reconstruction kernel, using image-to-image translation to convert archived CT to a kernel for which quantitative reference values exist[^tanabe2022kernel]. Deep-learning image reconstruction for dose reduction has also been shown to shift abdominal CT densitometry, which is handled in practice by correcting against a region of known attenuation[^vanstiphout2021]. Both lines treat the bias as a calibration problem with a known anchor. Slice thickness offers no such anchor, because the averaging acts on the lung parenchyma itself.

### Validity of a reconstruction without a reference

Generative reconstruction can synthesize structures that are absent from the measurement. The decomposition of an estimate into measurement-space and null-space components shows that such fabrications live in the null space of the forward operator, where data consistency is blind to them[^bhadra2021]. This result bounds what any consistency-based check can certify, and we use it to scope the indicator introduced below rather than to claim more of it than the theory permits.

## Proposed Approach

### Problem formulation

Let $x \in \mathbb{R}^{D \times H \times W}$ denote the 1 mm volume and $y \in \mathbb{R}^{(D/r) \times H \times W}$ the 5 mm volume reconstructed from the same acquisition, where $r = 5$ is the ratio of slice spacings and $D$, $H$, $W$ are the through-plane and in-plane dimensions. Thick-slice reconstruction applies a through-plane sensitivity profile followed by decimation. We model it as

$$y = A_w x + \varepsilon,$$

where $A_w$ convolves along the through-plane axis with a Gaussian of full width at half maximum $w$ and then averages each group of $r$ slices, and $\varepsilon$ collects acquisition noise. The effective width $w$ is a protocol constant, fitted once per protocol on calibration pairs.

Two properties of $A_w$ matter here. It is mean-preserving, so the mean attenuation of any slab is unchanged by the operator. And it has a non-trivial null space: through-plane detail that averages to zero within a slab leaves no trace in $y$.

The quantity to be recovered is not $x$ but a functional of it. For a lung set $\Omega$ and threshold $\tau$,

$$\mathrm{LAA}_\tau(x) = \frac{|\{v \in \Omega : x_v < \tau\}|}{|\Omega|},$$

and Perc15 is the corresponding quantile. Both depend on $x$ only through the histogram of $\{x_v : v \in \Omega\}$, and both are discontinuous in the sense that moves of the histogram across $\tau$ change them while moves elsewhere do not.

This is where a squared-error estimator and a densitometric estimator part company. Write the reconstruction as $\hat{x} = x + e$. The mean squared error is minimised by the conditional mean, which suppresses the high-frequency component of $e$ that the operator cannot constrain. Suppression of that component narrows the histogram, and a narrowed histogram has fewer voxels below $\tau$. The estimator can partially offset the resulting loss of LAA by shifting the whole histogram downward, and because the shift is a single scalar it is cheap in squared error relative to the tail mass it restores. The optimum of a deterministic regression therefore sits at a non-zero global displacement

$$\delta = \mathbb{E}_\Omega[\hat{x} - x],$$

and threshold statistics are the functionals most sensitive to $\delta$. We report $\delta$ as a measured quantity below.

### Restoration by residual sampling

Rather than regress $x$, we sample it. A base predictor supplies $x_0$, either through-plane upsampling of $y$ or a frozen restoration backbone. We work in the scaled residual

$$u = \frac{x - x_0}{s_r},$$

with $s_r$ a fixed scale, and learn a velocity field $v_\theta(u, t \mid y, w)$ by flow matching[^lipman2023], integrating from $t = 0$ to $t = 1$ with an explicit Euler scheme. Sampling rather than averaging preserves the width of the residual distribution, which is what the histogram sees, so the estimator has no reason to buy tail mass with a displacement.

After each integration step the iterate is projected back onto the set consistent with the measurement, which keeps $A_w \hat{x}$ close to $y$ while leaving the null-space component free for the sampler to populate.

### Reference-free competence indicator

Mean preservation gives an estimator of $\delta$ that needs no 1 mm reference. Because $A_w$ preserves slab means, any displacement of $\hat{x}$ appears undiminished in $A_w\hat{x} - y$ over the eroded lung set $\Omega_e$, so

$$\hat{\delta} = \frac{1}{|\Omega_e|}\sum_{v \in \Omega_e} \left(A_w\hat{x} - y\right)_v$$

is computable at deployment. Two further quantities accompany it. The structural residual $\rho_{\text{struct}} = \lVert (A_w\hat{x} - y) - \hat{\delta}\rVert / \lVert y - \bar{y}\rVert$ measures what remains after the displacement is removed, and $s$ compares the through-plane variance of $\hat{x}$ with the value predicted from $y$ by a calibration fitted on paired data, expressed in units of that calibration's held-out residual spread. Thresholds for all three come from the null distribution obtained by passing the true thin volume through the same estimators on calibration cases.

Two limits follow from the null-space argument[^bhadra2021]. All three quantities are computed from $A_w\hat{x} - y$ or from a variance predicted by $y$, so none of them sees the null space directly, and the indicator is accordingly used at the level of the scanner rather than the scan. We report it as such.

## Experimental Results

### Cohorts and endpoints

The training cohort comprised 203 paired examinations from one hospital. The validation cohort comprised 494 paired examinations from a second, independent hospital, of which 50 were used to fit protocol constants and indicator thresholds and 444 were reported. Validation examinations came from two manufacturers and two reconstruction kernels. A public paired cohort of 50 whole volumes was used for the comparison with methods whose published weights were available. Every index was computed on whole volumes over lung lobe masks from TotalSegmentator[^wasserthal2023], not over an attenuation window. Agreement is reported as bias, as 95% limits of agreement (LoA) in the sense of Bland and Altman[^bland1986], and as the concordance correlation coefficient (CCC) of Lin[^lin1989]. In the validation cohort the 1 mm reference gave LAA-950 7.82 ± 8.21%, LAA-910 19.72 ± 14.42%, and Perc15 −911.2 ± 43.3 HU.

### Reading the 5 mm series directly

Measured on the acquired 5 mm series with no reconstruction, LAA-950 read 2.83% against a reference of 7.82%, a bias of −4.99 percentage points with LoA of −13.50 to +3.51 and CCC 0.631 (Table 1). About two-thirds of the emphysema present was therefore not counted. Perc15 was overestimated by 25.7 HU and Perc10 by 29.9 HU (Table 3). Requiring all five sub-voxels of a thick voxel to be lung, rather than a majority, changed the mean absolute error from 5.05 to 5.02 percentage points, which locates the bias in through-plane partial-volume averaging rather than in the rule used to define the lung set.

### Image quality gains do not reach the measurement

Relative to reading the 5 mm series directly, Lanczos interpolation recovered 12% of the LAA-950 bias and a published restoration network recovered 10%, while the network exceeded interpolation by 2.0 dB in PSNR (Fig. 1d, Table 1). The two differed in LAA-950 bias by 0.07 percentage points. Across the four arms the relation between PSNR and densitometric bias was flat over a 3.2 dB span.

The public cohort showed the same ordering with far smaller absolute differences. Three published methods reached PSNR above 33 dB, about 3 dB above Lanczos, and all three had slightly larger LAA-950 bias than Lanczos (−0.82, −0.90 and −0.86 against −0.78 percentage points; Table 4). That cohort has a reference LAA-950 of 1.21 ± 1.14%, close to a normal population, so every method's error is small in absolute terms and the failure is invisible there. The difference in scale between the two cohorts is itself informative about why the gap has not been reported.

### A constant attenuation displacement

On the validation cohort the deterministic regression displaced the lung histogram by −16.32 ± 0.65 HU across 444 examinations (Fig. 1f, Table 5). A standard deviation of 0.65 HU on a −16 HU effect identifies the displacement as a constant of the estimator rather than case-level noise. On the public cohort, adding the densitometric indices to the training objective and varying their weight left the displacement within 0.01 HU of its original value, so the effect is not an artefact of the loss used. A control that applied a fitted scalar offset and nothing else displaced the histogram by −39.8 to −41.8 HU, confirming that a scalar correction alone is not a solution.

### Restored agreement

The residual sampler, applied zero-shot with weights trained only on public data, gave an LAA-950 bias of −0.26 percentage points with LoA of −5.61 to +5.08 and CCC 0.933, against −4.48 and 0.682 for the published network (Table 1). LAA-910 bias was −0.32 percentage points with CCC 0.986 (Table 2), and Perc15 bias was +2.5 HU against +19.3 HU (Table 3). Expressed against the unreconstructed series, the sampler recovered 95% of the LAA-950 bias. Its PSNR was 26.81 dB, below the 27.59 dB of the unreconstructed 5 mm series and 3.96 dB below the published network.

Agreement held regionally. Across the five lobes CCC ranged from 0.929 to 0.956 with biases between +0.84 and +1.50 percentage points (Table 6). Results were insensitive to the number of integration steps, with a paired absolute-error difference between 32 and 64 steps of 95% confidence interval −0.06 to +0.46 percentage points and a lung PSNR difference of 0.09 dB in 20 examinations.

Fine-tuning on paired data from the validation site tightened the LAA-950 limits of agreement to −3.18 to +5.26 and raised CCC to 0.948, and raised PSNR by 1.5 dB, at the cost of larger LAA-910 bias and larger percentile errors. Sites that can produce paired data can therefore trade one endpoint against another; sites without thin-slice capability cannot produce such data at all, which is why the zero-shot arm is the deployable one.

### Competence varies by scanner, not by scan

The indicator passed 80% of examinations in the domain the public model was trained on. On the validation cohort it passed 16.1% of the soft-kernel subset, where the densitometric endpoints held with a diagnostic reclassification rate of 1.6%, and 4.8% of the sharp-kernel subset, where they did not, with LAA-950 bias between −4 and −5.5 percentage points and a reclassification rate of 27.6% (Fig. 1g). After fine-tuning on matched data both scanner families passed at about 99% with endpoints intact. The sharp-kernel data lay inside the physical calibration set, so what the indicator separates is whether the model has seen this kind of data, not whether the protocol has seen this scanner.

Within a single scanner the verdict carried no case-level information. Among soft-kernel examinations, none of the 50 passing cases and 5 of the 261 flagged cases crossed the 10% diagnostic cut-off, a difference that is not significant (Fisher exact test, p = 1.0). We therefore report the indicator as a site-level gate and make no per-scan claim.

## Discussion

Image quality and densitometric accuracy came apart on this task, and the separation was not marginal. A published restoration network with 2.0 dB more PSNR than Lanczos interpolation carried the same LAA-950 bias to within 0.07 percentage points, and the method that restored agreement had the lowest PSNR of every arm tested, including the unreconstructed 5 mm series. Anyone selecting a restoration model on image quality alone would rank these methods in close to the reverse of their densitometric order.

The mechanism explains why. A deterministic estimator minimising squared error narrows the residual distribution it cannot constrain, and recovers part of the lost tail mass most cheaply by displacing the histogram. The displacement we measured, −16.32 HU with a standard deviation of 0.65 HU across 444 examinations, behaves as a property of the estimator rather than of the patient, and adding the densitometric indices to the training objective left it within 0.01 HU. Threshold statistics are precisely the functionals that such a displacement moves, which is why they fail while structural metrics do not. Sampling the residual rather than averaging it removes the incentive, and the displacement fell to −1.76 HU.

This scopes rather than contradicts the benchmark result that restoration improves downstream consistency[^yu2026benchmark]. Segmentation-derived measurements and radiomic features are shape and texture descriptors, largely invariant to a uniform attenuation shift, and there is no reason for them to degrade. Threshold densitometry is the endpoint that a shift moves by construction. Downstream endpoints are not interchangeable as validation, and the ones that stress a restoration hardest are those computed from the voxel histogram.

For sites without thin-slice capability, the practical reading is that reconstruction is worth doing and the choice of method matters more than its image quality suggests. Reading the 5 mm series directly missed about two-thirds of the emphysema present. Interpolation and a published network each recovered about a tenth of that bias; the residual sampler recovered 95%. The accompanying indicator addresses the question such a site must answer first, which is whether the model transfers to its scanner at all, and it answers that question without a thin-slice reference. It does not answer whether an individual scan can be trusted, and we have not presented it as though it does.

## Conclusion and Future Works

Deep-learning restoration of thin-slice chest CT improves image quality without improving emphysema densitometry, because the optimum of a deterministic estimator on this problem is a constant displacement of the attenuation histogram, and threshold indices are the quantities most sensitive to it. Sampling the reconstruction residual removes the displacement and recovers 95% of the bias present in the unreconstructed 5 mm series, on an external cohort of 444 examinations from two manufacturers, at an image quality below that of the series it started from. A reference-free indicator identifies scanners on which the model is not competent.

Three directions follow. The validation cohort covered two manufacturers and two kernels from one hospital, so the range of protocols over which the fitted operator and the indicator thresholds transfer is not yet mapped; multi-site evaluation would establish it, and the indicator provides a way to screen a new site before any reference exists. The endpoints here are agreement with 1 mm densitometry rather than clinical outcome, so a reading study with radiologists, and comparison against pulmonary function or longitudinal decline, are needed to establish that the recovered agreement changes what a clinician concludes. Finally, the sampler produced non-finite values on four of 444 examinations whose lung masks were normal, which the indicator now abstains on rather than passing; the conditions under which the integration diverges deserve characterisation before deployment at scale.

## Data availability

The public paired cohort is available from the RPLHR-CT release[^yu2022tvsrn]. The clinical cohorts contain identifiable imaging and are held under institutional agreements that do not permit redistribution; derived per-examination measurements supporting all reported statistics are available from the corresponding author on reasonable request, subject to approval by the contributing institutions.

## Code availability

Training, inference, calibration, and evaluation code, together with the model weights used for the zero-shot arm, are available at the project repository.

[^lynch2015]: Lynch, D. A. et al. Radiology 277, 192–205 (2015).
[^gierada2010]: Gierada, D. S. et al. Acad. Radiol. 17, 146–156 (2010).
[^yu2024cthnet]: Yu, P. et al. npj Digit. Med. 7, 335 (2024).
[^yu2026benchmark]: Yu, P. et al. npj Digit. Med. (2026).
[^yu2022tvsrn]: Yu, P. et al. MICCAI, 344–353 (2022).
[^song2024i3net]: Song, H. et al. IEEE Trans. Med. Imaging 43, 3306–3318 (2024).
[^tanabe2022kernel]: Tanabe, N. et al. Front. Artif. Intell. 4, 769557 (2022).
[^vanstiphout2021]: van Stiphout, J. A. et al. Eur. Radiol. 32, 2921–2929 (2021).
[^bhadra2021]: Bhadra, S. et al. IEEE Trans. Med. Imaging 40, 3249–3260 (2021).
[^lipman2023]: Lipman, Y. et al. ICLR (2023).
[^wasserthal2023]: Wasserthal, J. et al. Radiol. Artif. Intell. 5, e230024 (2023).
[^bland1986]: Bland, J. M. & Altman, D. G. Lancet 327, 307–310 (1986).
[^lin1989]: Lin, L. I.-K. Biometrics 45, 255–268 (1989).
