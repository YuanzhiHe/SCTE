# CT-DegradBench: A Physics-Informed Benchmark for CT Degradation Detection and Severity Estimation

paper_id: arxiv:2605.16431v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Computed tomography (CT) is a core medical imaging modality that provides high-resolution anatomical information essential for clinical diagnosis, treatment planning, and image-guided intervention
13
;
4
. In routine practice, however, CT images are often degraded by artifacts introduced during acquisition and reconstruction, including noise, blur, streaking, aliasing, and metal artifacts
4
;
19
;
28
;
11
. These degradations arise from diverse sources, such as dose reduction, sparse-view sampling, metallic implants, patient motion, and reconstruction parameter choices, and they frequently occur at different severity levels or in combination. Such effects compromise visual interpretation and negatively impact downstream automated analysis
18
.
A large body of work has been devoted to CT image restoration, with deep learning methods now dominating low-dose denoising, sparse-view reconstruction, metal artifact reduction, and related enhancement tasks
4
;
11
;
28
;
17
;
16
;
22
;
7
;
23
. Yet despite this progress, evaluation still depends heavily on image quality assessment (IQA) metrics whose perceptual and clinical reliability across diverse degradation conditions remains limited
26
;
10
;
4
;
11
. At the same time, most existing datasets are designed for isolated restoration settings, such as low-dose denoising, sparse-view reconstruction, or metal artifact reduction, rather than for unified analysis across multiple degradation types and severity levels. As a result, they provide limited support for systematic benchmarking of degradation detection, severity estimation, and IQA reliability under controlled conditions.
This exposes an important gap: while CT restoration has been widely studied, there is still no unified benchmark for analyzing how different degradations and severity levels affect CT image quality, nor for assessing whether commonly used metrics and modern representation models remain sensitive and reliable across such conditions. This problem is especially relevant in practice, where degradations rarely appear in a single canonical form and often co-occur in compound scenarios.
To address this gap, we introduce
CT-DegradBench
, a dataset and benchmark for CT artifact detection and severity estimation under controlled single- and mixed-degradation settings. CT-DegradBench is built from paired reference and degraded CT images generated using physics-informed degradation models and covers five common acquisition-related degradation types across calibrated severity levels. It also includes radiologist-informed mixed-artifact settings to approximate clinically plausible compound degradations.
Figure 1
:
CT-DegradBench generation pipeline.
A reference CT image is forward-projected to the sinogram domain, where physics-informed degradations are applied individually or as realistic mixtures with controlled severity. The degraded sinogram is reconstructed via filtered backprojection to obtain the final degraded CT image, while structured metadata are generated in parallel for prompt construction.
Using CT-DegradBench, we first conduct a systematic benchmark of classical and deep learning-based IQA metrics commonly used in CT enhancement evaluation, with a focus on degradation sensitivity and severity monotonicity. We then extend the benchmark to modern multimodal models by studying whether medical vision-language models (VLMs) encode quality-aware representations that correlate with degradation type and severity.
Building on this analysis, we propose
SeSpeCT
(
Se
mantic-
Spe
ctral
CT
degradation estimation), a framework for joint artifact detection and severity estimation. SeSpeCT constructs a training-free semantic quality axis in the multimodal embedding space of medical VLMs using radiology-informed text prompts, without task-specific fine-tuning, and combines this semantic signal with complementary spectral features that capture degradation-specific frequency structure. Together, these components provide a robust representation for CT degradation analysis under both isolated and mixed-artifact conditions.
Our contributions are summarized as follows:
•
We introduce
CT-DegradBench
, a controlled benchmark for CT degradation detection and severity estimation, covering five common degradation types, calibrated severity levels, and radiologist-informed mixed-artifact settings.
•
We provide the first
systematic evaluation of CT degradation sensitivity
across classical IQA metrics, learned perceptual metrics, and medical vision-language models, analyzing their behavior under diverse single- and mixed-artifact conditions.
•
We propose
SeSpeCT
, a semantic-spectral framework that combines a training-free prompt-derived quality axis from medical vision-language models with frequency-domain features for joint artifact type and severity prediction.

## Method

Existing CT datasets are largely designed for task-specific restoration problems, such as denoising or metal artifact reduction, and therefore do not support unified benchmarking across degradation types, severity levels, and mixed-artifact settings. We address this limitation with
CT-DegradBench
, a controlled multi-degradation CT dataset and benchmark for artifact detection and severity estimation. CT-DegradBench generates paired
reference–degraded
CT slices with explicit degradation labels, calibrated severity levels, and clinically plausible artifact mixtures, enabling systematic evaluation under a common experimental framework.
2.1
Problem setting and dataset overview
Let
x
ref
∈
ℝ
H
×
W
x^{\mathrm{ref}}\in\mathbb{R}^{H\times W}
denote a reference CT image. We generate a degraded counterpart
x
deg
x^{\mathrm{deg}}
by applying one or more degradation operators
𝒟
k
​
(
⋅
)
\mathcal{D}_{k}(\cdot)
with known severity:
x
deg
=
𝒟
K
∘
⋯
∘
𝒟
2
∘
𝒟
1
(
x
ref
)
,
x^{\mathrm{deg}}=\mathcal{D}_{K}\circ\cdots\circ\mathcal{D}_{2}\circ\mathcal{D}_{1}\left(x^{\mathrm{ref}}\right),
(1)
where each operator corresponds to a degradation type, namely noise, loss of sharpness, streaking, aliasing, or metal artifacts, and is associated with one of four severity levels (L0–L3), where L0 denotes the lowest severity level and L3 the highest. Each generated sample is accompanied by structured metadata specifying the degradation type(s), severity level(s), mixture composition, and generation parameters. This formulation enables controlled construction of both isolated and mixed degradations, making CT-DegradBench suitable for standardized benchmarking of degradation detection, severity estimation, and quality-aware representation learning.
2.2
CT imaging background
Computed tomography reconstructs cross-sectional images from X-ray projections acquired at multiple view angles. In clinical CT, reconstructed images are commonly represented in Hounsfield Units (HU), which encode attenuation relative to water. Given a reference CT slice
x
ref
x^{\mathrm{ref}}
in HU, we convert it to linear attenuation coefficients as
μ
=
μ
w
​
(
1
+
x
ref
1000
)
,
\mu=\mu_{\mathrm{w}}\left(1+\frac{x^{\mathrm{ref}}}{1000}\right),
(2)
where
μ
w
\mu_{\mathrm{w}}
denotes the attenuation coefficient of water
9
.
The acquisition process is modeled in the projection domain. An ideal sinogram
s
s
is obtained by applying the parallel-beam Radon transform
s
=
ℛ
⁡
(
μ
)
.
s=\mathcal{R}(\mu).
(3)
Image reconstruction is then performed using filtered backprojection (FBP)
2
:
μ
^
=
ℛ
FBP
−
1
​
(
s
)
,
x
^
=
1000
​
(
μ
^
μ
w
−
1
)
,
\hat{\mu}=\mathcal{R}^{-1}_{\mathrm{FBP}}(s),\hskip 20.00003pt\hat{x}=1000\left(\frac{\hat{\mu}}{\mu_{\mathrm{w}}}-1\right),
(4)
where
x
^
\hat{x}
is the reconstructed CT image in HU.
This formulation provides a physics-grounded basis for degradation simulation. In CT-DegradBench, degradations are introduced prior to reconstruction, either directly in the projection domain or by modifying the attenuation map in the image domain (e.g., for metal artifacts), and are subsequently propagated to the image domain through filtered backprojection.
2.3
Degradation modeling and severity levels
To enable controlled benchmarking, we model five common CT degradations, noise, loss of sharpness, streaking, aliasing, and metal artifacts, each at four calibrated severity levels (L0–L3).
(1) Noise (mixed Poisson–Gaussian, severity
γ
\gamma
).
CT noise arises mainly from photon counting statistics and electronic detector noise in low-dose CT imaging. We simulate it in the projection domain using the mixed Poisson–Gaussian model of
29
. Given a clean sinogram
s
s
, photon counts are sampled as
K
∼
Poisson
⁡
(
α
​
I
0
​
e
−
s
)
,
K\sim\mathrm{Poisson}\!\left(\alpha I_{0}e^{-s}\right),
(5)
where
I
0
I_{0}
is the incident photon intensity and
α
\alpha
controls the effective dose level. Detector noise is modeled by
K
′
=
K
+
𝒩
⁡
(
0
,
σ
2
)
,
K^{\prime}=K+\mathcal{N}(0,\sigma^{2}),
(6)
and the noisy sinogram is obtained via
s
noisy
=
−
log
⁡
(
K
′
+
δ
α
​
I
0
)
,
s_{\mathrm{noisy}}=-\log\!\left(\frac{K^{\prime}+\delta}{\alpha I_{0}}\right),
(7)
where
δ
\delta
stabilizes the logarithm.
To generate controlled degradation levels for benchmarking, we introduce a residual noise scaling mechanism that enables calibrated control of noise severity while preserving noise characteristics consistent with the underlying CT acquisition physics.
Specifically, the residual noise is scaled as
s
noisy
γ
=
s
+
γ
⁡
(
s
noisy
−
s
)
,
γ
∈
{
1
,
2
,
2.5
,
4
}
.
s_{\mathrm{noisy}}^{\gamma}=s+\gamma\left(s_{\mathrm{noisy}}-s\right),\hskip 20.00003pt\gamma\in\{1,\ 2,\ 2.5,\ 4\}.
(8)
The degraded sinogram is then reconstructed using FBP (Eq.
4
).
(2) Loss of sharpness / blur (severity
σ
\sigma
).
The dominant resolution loss in CT arises from detector aperture and detector response, which primarily affect spatial resolution along the detector direction rather than across projection angles. Accordingly, we blur the sinogram along the detector axis with a 1D Gaussian kernel:
s
blur
=
G
σ
∗
s
,
σ
∈
{
0.8
,
1
,
1.5
,
2.5
}
,
s_{\text{blur}}=G_{\sigma}*s,\hskip 10.00002pt\sigma\in\{0.8,1,1.5,2.5\},
(9)
where
∗
*
denotes convolution and
G
σ
G_{\sigma}
is applied along the detector dimension. The blurred sinogram is then reconstructed using FBP.
(3) Streaks ( severity
Δ
​
L
\Delta L
).
In practice, streak artifacts often co-occur with noise and other degradations. To isolate their effect, we introduce structured outliers in the sinogram that mimic directional streaks observed in clinical CT images:
s
streak
=
s
+
Δ
​
L
⋅
M
,
Δ
​
L
∈
{
0.25
,
0.5
,
1
,
2
}
,
s_{\mathrm{streak}}=s+\Delta L\cdot M,\hskip 20.00003pt\Delta L\in\{0.25,\ 0.5,\ 1,\ 2\},
(10)
Figure 2
:
Overview of the proposed SeSpeCT framework.
The model combines a semantic quality branch derived from a medical vision–language model with frequency-domain descriptors extracted from the Fourier spectrum.
The fused representation is used to jointly predict degradation type and severity.
where
M
M
is a binary mask selecting the affected regions. Reconstructing
s
streak
s_{\mathrm{streak}}
yields the streak-corrupted image.
(4) Aliasing artifacts (severity: number of views).
Aliasing artifacts from sparse-view CT
7
are simulated by subsampling projection angles. Let
Θ
\Theta
denote the full set of acquisition angles and
Θ
N
⊂
Θ
\Theta_{N}\subset\Theta
a subset of
N
N
views. We form a sparse sinogram
s
N
=
ℛ
Θ
N
​
(
μ
)
,
N
∈
{
180
,
90
,
60
,
45
}
,
s_{N}=\mathcal{R}_{\Theta_{N}}(\mu),\hskip 20.00003ptN\in\{180,\ 90,\ 60,\ 45\},
(11)
and reconstruct it using FBP. Fewer views produce stronger aliasing artifacts.
(5) Metal artifacts.
Metal artifacts are simulated by inserting a metal region into the attenuation map, following
8
. Given a binary metal mask
m
⁡
(
𝐱
)
∈
{
0
,
1
}
m(\mathbf{x})\in\{0,1\}
and metal attenuation coefficient
μ
metal
\mu_{\mathrm{metal}}
, we define
μ
metal
​
(
𝐱
)
=
(
1
−
m
⁡
(
𝐱
)
)
​
μ
​
(
𝐱
)
+
m
⁡
(
𝐱
)
​
μ
metal
.
\mu_{\mathrm{metal}}(\mathbf{x})=(1-m(\mathbf{x}))\,\mu(\mathbf{x})+m(\mathbf{x})\,\mu_{\mathrm{metal}}.
(12)
We then forward-project and reconstruct using Eq.
4
, yielding artifacts similar to those observed around metallic implants. Severity is controlled by increasing the size of the metal region. Because metal artifacts are spatially localized, we also provide bounding-box annotations derived from the mask
m
⁡
(
𝐱
)
m(\mathbf{x})
.
Mixed degradations.
In clinical CT, multiple artifacts often co-occur due to interacting acquisition conditions, reconstruction processes, and patient-specific factors. To reflect this setting, CT-DegradBench includes mixed degradations in addition to isolated artifacts. We define five representative mixture configurations:
Blur + Noise
,
Streaks + Noise
,
Metal + Noise
,
Aliasing + Noise
, and
Metal + Blur + Noise
.
These combinations capture common interactions between acquisition noise and other degradation mechanisms. Degradations are applied sequentially in an order that approximates their manifestation in reconstructed CT images; the corresponding clinical scenarios are summarized in the supplementary material (Table
8
).
Each mixture is assigned a global severity level controlling the overall degradation strength. Component severities are sampled from neighboring levels around this global level to maintain comparable artifact magnitudes and avoid unrealistic combinations. We define the final mixture severity as the maximum severity among its constituent degradations.
2.4
Metadata Description
To support reproducible benchmarking, each CT-DegradBench sample is associated with structured metadata describing the degradation type(s), mixture order, severity level, and generation parameters. For metal artifacts, localization annotations are additionally provided. A structured natural-language prompt describing the degradation configuration is included to benchmark Vision–Language Models (VLMs), as illustrated in Figure
1
.
