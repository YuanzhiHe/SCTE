# Deep Slice Interpolation for Reducing Through-Plane Anisotropy and Noise in Head CT

paper_id: semanticscholar:b09c6588b37ff104bf84a6a1ed1e0fe895eeed43
tier: T2
source_used: html_arxiv
warning: none

## Intro

Head computed tomography (CT) volumes are frequently reconstructed with markedly anisotropic voxel spacing, typically exhibiting sub-millimeter in-plane resolution but 2–5 mm through-plane spacing, a reconstruction convention that improves per-slice signal-to-noise ratio and matches established reading workflows. This anisotropy degrades continuity in multiplanar visualizations, introduces stair-step artifacts in coronal and sagittal views, and adversely affects both visual assessment and downstream pipelines that assume near-isotropic sampling, including volumetric measurements such as hematoma quantification and automated triage systems. Reducing the effective slice spacing through interpolation, if not done during image acquisition or reconstruction, can therefore improve three-dimensional visualization for surgical planning, quantitative analysis, and the reliability of learning-based clinical tools.
We present a deep learning system for head CT slice interpolation that synthesizes an intermediate axial slice
I
k
+
1
I_{k+1}
from two neighboring slices
(
I
k
,
I
k
+
2
)
(I_{k},I_{k+2})
. The system uses a U-Net with an EfficientNetV2-S encoder, trained on the RSNA 2019 Intracranial Hemorrhage Detection dataset
[
9
]
, which was used in prior work on intracranial hemorrhage detection
[
6
]
. The scope of this work is restricted to non-contrast head CT: the training data, Hounsfield unit (HU) window, and target through-plane spacing are all chosen for this setting, and we make no claim of generalization to other anatomical regions, other modalities, or different acquisition protocols. Cross-anatomy transfer is discussed explicitly as a limitation (
Section
5.7
) and identified as future work.
The contributions of this work are:
1.
A
deep learning system for head CT slice interpolation
that outperforms classical baselines and pretrained video frame interpolation models on all structural metrics. Video frame interpolation is a natural transfer source because it addresses the same core problem (synthesizing an intermediate image from two neighboring observations), albeit in the temporal rather than spatial domain; we benchmark RIFE
[
14
]
and FILM
[
29
]
to quantify the domain gap.
2.
A
systematic loss function evaluation
under a fixed architecture and training protocol, providing practical guidance on which losses and settings are reliable or unreliable. As part of this analysis, we identify systematic training divergence in SSIM-family losses traceable to numerical conditioning of the SSIM denominator, and provide actionable remedies.
3.
A
finding that regression-based synthesis inherits the denoising property
of conditional-expectation estimation
[
2
,
12
]
, as leveraged for image restoration from noisy targets by Lehtinen
et al.
[
24
]
, connecting slice interpolation to the low-dose CT denoising literature and delivering interpolation and implicit denoising from a single model.
The remainder of this paper is organized as follows.
Section
2
reviews through-plane CT super-resolution, video frame interpolation, loss-function comparisons, and low-dose CT denoising.
Section
3
describes the dataset, the U-Net with EfficientNetV2-S architecture, the eleven loss configurations evaluated, and the patient-level statistical protocol.
Section
4
reports held-out test performance with bootstrap confidence intervals and paired Wilcoxon tests, together with a quantitative ROI-level measurement of implicit denoising.
Section
5
analyzes loss–metric alignment, SSIM training instability, implicit denoising as a consequence of regression-based synthesis, an out-of-distribution case study, the effect of pathology on interpolation difficulty, and limitations.
Section
6
summarizes the findings and outlines directions for future work.

## Method

3.1
Dataset and preprocessing
We use the RSNA Intracranial Hemorrhage Detection dataset
[
9
]
, which provides slice-level multi-label annotations for five hemorrhage subtypes (epidural, subdural, subarachnoid, intraparenchymal, intraventricular) plus an
any
-hemorrhage flag. These categories are RSNA challenge labels rather than ICD-10/ICD-11 hemorrhage subtypes. The dataset is distributed through the Kaggle competition platform under the competition’s data-use agreement; our training code, model configurations, and evaluation scripts are released separately (see
Appendix
A
). The RSNA dataset is multi-institutional with heterogeneous acquisition protocols; reported inter-slice spacing typically falls in the 2.5–5 mm range with sub-millimeter in-plane resolution. Synthesizing one intermediate slice between each pair of acquired slices halves the effective through-plane spacing from
∼
5
{\sim}5
mm to
∼
2.5
{\sim}2.5
mm.
DICOM slices are windowed with center
44
44
HU and width
128
128
HU. We clamp the upper bound at
107
107
HU so that the windowed interval covers
128
128
integer HU steps and discretizes uniformly to the
256
256
levels of
8
8
-bit PNG storage (two levels per HU); the effective HU window is therefore
[
−
20
​
HU
,
107
​
HU
]
[-20\,\text{HU},107\,\text{HU}]
. The windowed slices are converted to grayscale PNG and normalized to
[
0
,
1
]
[0,1]
, so that
−
20
​
HU
-20\,\text{HU}
maps to
0
0
and
107
​
HU
107\,\text{HU}
to
1
1
. All training, evaluation, and interpolation operate on these
[
0
,
1
]
[0,1]
-normalized intensities; for clarity we report intensity ranges in HU units throughout the paper. Values outside
[
−
20
​
HU
,
107
​
HU
]
[-20\,\text{HU},107\,\text{HU}]
are saturated at preprocessing, and the model is neither trained nor calibrated for out-of-window content.
The prior hemorrhage detection work
[
6
]
used three HU windows stacked as RGB channels. For the single-channel interpolation task, we selected this window to cover brain parenchyma (
∼
20
{\sim}20
–
45
45
HU) and acute hemorrhage (
∼
60
{\sim}60
–
80
80
HU); it is wider than the standard brain window (center
40
40
HU, width
80
80
HU, range
[
0
​
HU
,
80
​
HU
]
[0\,\text{HU},80\,\text{HU}]
) and leaves headroom on both sides of the clinically relevant intensities. Air (below
−
20
-20
HU) and bone (above
107
107
HU) still fall outside the window.
We use a standard patient-level split into training, validation, and test sets: 30 patients are held out as the test set, and the remaining 18,870 patients are split 80/20 into training (15,096 patients, 570,562 triplets) and validation (3,774 patients, 141,747 triplets).
9.3
%
9.3\%
of RSNA patients (
1,755
1{,}755
of
18,900
18{,}900
) are associated with multiple CT studies; we treat each study as an independent sequence when constructing triplets, so no triplet spans two studies. The patient-level split assigns all studies of a given patient to a single partition, and the 30 held-out test patients are additionally restricted to those with exactly one study so that patient-level paired comparisons are unambiguous.
The 30 test patients were selected to enable the patient-level paired tests reported in
Section
4.2
. From the subset of patients with exactly one CT study, patients were selected to enrich the test set for pathology while guaranteeing representation of every hemorrhage subtype: for each of the 5 subtypes, 5 patients containing that subtype were randomly selected (without replacement across subtypes) and removed from the candidate pool; 5 additional patients were then drawn uniformly at random from the remaining pool, which contained both hemorrhage and normal studies. This yielded 27 patients carrying at least one hemorrhage subtype and 3 purely normal patients, totaling 968 interpolation triplets. The pathology enrichment is intentional: the clinically motivated question is whether interpolation quality holds on abnormal anatomy, where structural fidelity matters most; the 3 normal patients (572 slice-level triplets) provide a sanity check that performance does not degrade on normal cases. At the slice level, 396 triplets (41%) contain hemorrhage and 572 (59%) do not; this hemorrhage prevalence is substantially higher than in unselected head-CT cohorts. The test-set patient IDs are released as a frozen artefact. Hemorrhage labels are multi-label (190 of 396 triplets carry more than one subtype), so “any-of” counts exceed the total: subdural (
n
=
189
n=189
), subarachnoid (
n
=
184
n=184
), intraparenchymal (
n
=
132
n=132
), intraventricular (
n
=
114
n=114
), and epidural (
n
=
44
n=44
); these per-subtype counts support the hemorrhage-stratified analysis of
Section
4.3
.
3.2
Task definition
Given consecutive triplets
(
I
k
,
I
k
+
1
,
I
k
+
2
)
(I_{k},I_{k+1},I_{k+2})
, input is
X
=
[
I
k
,
I
k
+
2
]
X=[I_{k},I_{k+2}]
(2 channels), target is
Y
=
I
k
+
1
Y=I_{k+1}
, and the model’s prediction is denoted
Y
^
\hat{Y}
. This formulation assumes that the target slice is equidistant from the two input slices. The RSNA dataset is largely acquired at a single fixed inter-slice spacing per patient, so this assumption holds for the majority of cases; a small number of test patients contain variable-spacing series, for which the equidistance assumption is approximate.
3.3
Model and augmentation
Input
2
×
256
×
256
2\times 256\times 256
Stage 1
24
×
128
×
128
24\times 128\times 128
Stage 2
48
×
64
×
64
48\times 64\times 64
Stage 3
64
×
32
×
32
64\times 32\times 32
Stage 4
160
×
16
×
16
160\times 16\times 16
Stage 5 (bottleneck)
256
×
8
×
8
256\times 8\times 8
Decoder 5
+
3
×
3
+\;3{\times}3
conv
1
×
256
×
256
1\times 256\times 256
Decoder 4
32
×
128
×
128
32\times 128\times 128
Decoder 3
64
×
64
×
64
64\times 64\times 64
Decoder 2
128
×
32
×
32
128\times 32\times 32
Decoder 1
256
×
16
×
16
256\times 16\times 16
160
64
48
24
EfficientNetV2-S Encoder
U-Net Decoder
[
I
k
,
I
k
+
2
]
[I_{k},\;I_{k+2}]
I
^
k
+
1
\hat{I}_{k+1}
Figure 1
:
U-Net + EfficientNetV2-S architecture for CT slice interpolation.
Top row: encoder (blue), left-to-right in decreasing spatial resolution.
Bottom row: decoder (orange), right-to-left in increasing spatial
resolution. Columns are aligned by spatial resolution: the leftmost
column carries the 2-channel input
[
I
k
,
I
k
+
2
]
[I_{k},I_{k+2}]
and the final
decoder block fused with its
3
×
3
3\times 3
conv output projection
(producing the single-channel prediction
I
^
k
+
1
\hat{I}_{k+1}
); the next four
columns pair each encoder stage with the decoder block that consumes its
skip connection (dashed, annotated with the skip channel count); the
rightmost column carries the shared bottleneck. The absence of a skip
arrow in the leftmost column reflects the fact that the final decoder
block has no encoder skip. All dimensions shown correspond to the
256
×
256
256\times 256
patch resolution used at both training and inference
time; at test time, 13 such patch predictions are reassembled into a
512
×
512
512\times 512
composite (see
Section
3.3
).
We hold the network architecture fixed across all experiments in order to isolate the impact of training design choices. Accordingly, we do not claim universality of the reported findings across different architectural designs.
The architecture is a U-Net
[
31
]
with an EfficientNetV2-S encoder
[
34
]
(
Figure
1
); implementation libraries and versions are listed in
Appendix
A
. The encoder produces features at four skip-connection levels (channel dimensions
24
,
48
,
64
,
160
24,48,64,160
at strides
2
,
4
,
8
,
16
2,4,8,16
) plus a
256
256
-channel bottleneck at stride
32
32
. The bottleneck feeds the decoder, and the four skip-connection levels are concatenated into the corresponding decoder stages. The decoder consists of five upsampling blocks with output channels
(
256,128
,
64
,
32
,
16
)
(256,128,64,32,16)
; each block doubles the spatial resolution, concatenates the upsampled feature map with its encoder skip (the final block, operating at full resolution, has no skip), and applies two
3
×
3
3\times 3
Conv–BatchNorm–ReLU sub-blocks. A final
3
×
3
3\times 3
convolution maps the 16-channel feature map to a single-channel output, with no output activation applied: the task is regression, and the target range
[
0
,
1
]
[0,1]
is controlled by the normalized data and the bounded losses of
Section
3.4
. The encoder is initialized from ImageNet-pretrained weights; the decoder and segmentation head are initialized from scratch: decoder convolutions use Kaiming-uniform weight initialization for ReLU
[
13
]
(fan-in mode, zero bias), batch normalization layers are set to (weight
=
1
=1
, bias
=
0
=0
), and the final
3
×
3
3\times 3
head uses Xavier-uniform weight initialization
[
11
]
with zero bias. Because ImageNet pretraining uses 3-channel RGB input while our task has 2-channel input
(
I
k
,
I
k
+
2
)
(I_{k},I_{k+2})
, the first convolutional layer (stem) is adapted by keeping the first two of the three pretrained RGB filters (dropping the blue channel) and rescaling them by
3
/
2
3/2
to approximately preserve the activation magnitude at the stem output.
Table
1
summarizes the model’s computational profile.
Table 1
:
Model complexity (U-Net + EfficientNetV2-S). Training and test-time inference both operate on
256
×
256
256\times 256
patches; each test prediction is reassembled from 13 patches (9 overlapping center crops + 4 corner patches) into a
512
×
512
512\times 512
composite (see
Section
3.3
). GFLOPs are reported as
2
×
2\times
multiply–accumulate operations (the convention used by fvcore and ptflops).
Property
Value
Total parameters
22.1 M
Encoder (pretrained)
19.8 M
Decoder (random init)
2.2 M
Segmentation head (
×
3
3\!\times\!3
conv)
<
<
0.1 M
GFLOPs per
256
×
256
256\times 256
patch
12.4
GFLOPs per reconstructed slice (13 patches)
161
Figure
2
shows the patch geometry shared by training-time sampling and test-time inference. Each training sample draws one
256
×
256
256\times 256
view from a discrete pool: the nine overlapping center crops of
Figure
2
(a), plus a full-slice Lanczos
[
7
]
downsample to
256
×
256
256\times 256
(not shown in the figure). One of these ten options is drawn per training sample with weighted random sampling: the central crop carries twice the weight of each off-center crop, and the Lanczos option carries half. Normalizing the weights to sum to
1
1
yields selection probabilities of
≈
0.19
\approx\!0.19
for the central crop,
≈
0.095
\approx\!0.095
for each of the eight off-center crops, and
≈
0.048
\approx\!0.048
for the Lanczos downsample. After the spatial pick we apply an independent
50
%
50\%
horizontal flip,
50
%
50\%
vertical flip, and a
50
%
50\%
chance of rotation uniformly drawn from
[
−
15
∘
,
+
15
∘
]
[-15^{\circ},+15^{\circ}]
; all three transforms are applied identically to
I
k
I_{k}
,
I
k
+
1
I_{k+1}
, and
I
k
+
2
I_{k+2}
to preserve triplet alignment. At test time, each
512
×
512
512\times 512
slice is reconstructed from 13
256
×
256
256\times 256
patches: the same nine overlapping center crops (
Figure
2
(a)) are averaged over their overlaps to produce the central
384
×
384
384\times 384
band, and four non-overlapping corner patches
C
1
C_{1}
–
C
4
C_{4}
(
Figure
2
(b)) tile the full slice; the averaged nine-crop band is then overlaid on top, so only the outer
64
64
-pixel strip of each corner patch contributes to the final composite (
Figure
2
(c)). The composite is compared against the
512
×
512
512\times 512
target, preserving direct comparability with the classical and VFI baselines which produce full-resolution predictions natively. Validation drives early stopping and best-epoch checkpoint selection; it evaluates the model on each of the nine
256
×
256
256\times 256
center crops and averages the resulting metrics.
512
×
512
512\times 512
(a) 9 overlapping center crops
512
×
512
512\times 512
C1
C2
C3
C4
(b) 4 corner patches
512
×
512
512\times 512
central
384
×
384
384\times 384
from (a)
C1
C2
C3
C4
(c) Composite: (a) over (b), 13 patches
Figure 2
:
Patch geometry shared by training-time sampling and test-time
reconstruction. (a) Nine overlapping
256
×
256
256\times 256
crops whose
top-left corners form a
3
×
3
3\times 3
grid at 64-pixel stride in the
512
×
512
512\times 512
slice; their union covers the central
384
×
384
384\times 384
band (solid blue outline). Translucent fills overlap so deeper color
indicates regions covered by more crops; the dashed outline marks one
representative crop (top-left of the grid) to indicate the
256
×
256
256\times 256
scale of a single view. At training time one crop is
drawn per sample; at test time their predictions are averaged over
overlaps to reconstruct the central band. (b) Four non-overlapping
256
×
256
256\times 256
corner patches
C
1
C_{1}
–
C
4
C_{4}
with top-left corners at
(
y
,
x
)
=
(
0
,
0
)
(y,x)=(0,0)
,
(
0,256
)
(0,256)
,
(
256
,
0
)
(256,0)
,
(
256,256
)
(256,256)
, tiling the full
512
×
512
512\times 512
slice. (c) Test-time
composite: the central
384
×
384
384\times 384
band from (a) is placed on top
of the four corner patches from (b), yielding a 13-patch
512
×
512
512\times 512
prediction that is directly comparable to the classical and VFI
baselines.
3.4
Loss families and numerical stability
We evaluate four single-term losses (two pixel-wise and two structural) and two combined-loss families (SSIM+L1 and MS-SSIM+L1) whose mixing weight
α
\alpha
is swept across configurations. All losses operate on predictions
Y
^
\hat{Y}
and targets
Y
Y
normalized to
[
0
,
1
]
[0,1]
and indexed over
N
N
pixels.
Pixel-wise losses.
The two pixel-wise losses are the mean squared error and the mean absolute error,
ℒ
MSE
(
Y
^
,
Y
)
=
1
N
∑
i
=
1
N
(
y
^
i
−
y
i
)
2
,
ℒ
L1
(
Y
^
,
Y
)
=
1
N
∑
i
=
1
N
|
y
^
i
−
y
i
|
.
\mathcal{L}_{\mathrm{MSE}}(\hat{Y},Y)\;=\;\frac{1}{N}\sum_{i=1}^{N}\bigl(\hat{y}_{i}-y_{i}\bigr)^{2},\qquad\mathcal{L}_{\mathrm{L1}}(\hat{Y},Y)\;=\;\frac{1}{N}\sum_{i=1}^{N}\bigl\lvert\hat{y}_{i}-y_{i}\bigr\rvert.
(1)
Both are standard regression losses whose minimizers are well-known conditional statistics:
ℒ
MSE
\mathcal{L}_{\mathrm{MSE}}
has the conditional mean
𝔼
⁡
[
Y
∣
X
]
\mathbb{E}[Y\mid X]
as its minimizer, and
ℒ
L1
\mathcal{L}_{\mathrm{L1}}
the conditional median
[
2
,
12
]
; this property is what underpins the implicit-denoising argument developed in
Section
5.4
. The theoretical argument applies to these pixel-wise losses specifically; SSIM-based losses are not conditional-mean or -median estimators and so fall outside it, but are nonetheless measured to denoise empirically at reduced strength (
Section
4.4
).
Structural losses.
The single-scale structural similarity index (SSIM)
[
36
]
is computed on sliding
11
×
11
11\times 11
Gaussian-weighted windows with
σ
=
1.5
\sigma=1.5
. For aligned windows
x
,
y
x,y
with local means
μ
x
,
μ
y
\mu_{x},\mu_{y}
, variances
σ
x
2
,
σ
y
2
\sigma_{x}^{2},\sigma_{y}^{2}
, and covariance
σ
x
​
y
\sigma_{xy}
,
SSIM
⁡
(
x
,
y
)
=
(
2
​
μ
x
​
μ
y
+
C
1
)
​
(
2
​
σ
x
​
y
+
C
2
)
(
μ
x
2
+
μ
y
2
+
C
1
)
​
(
σ
x
2
+
σ
y
2
+
C
2
)
,
C
k
=
(
K
k
​
L
)
2
,
\mathrm{SSIM}(x,y)\;=\;\frac{\bigl(2\mu_{x}\mu_{y}+C_{1}\bigr)\bigl(2\sigma_{xy}+C_{2}\bigr)}{\bigl(\mu_{x}^{2}+\mu_{y}^{2}+C_{1}\bigr)\bigl(\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}\bigr)},\qquad C_{k}\;=\;(K_{k}L)^{2},
(2)
where
L
=
1
L=1
is the dynamic range of our
[
0
,
1
]
[0,1]
-normalized intensities (
Section
3.1
) and
(
K
1
,
K
2
)
(K_{1},K_{2})
are small dimensionless scalars whose only role is to keep the denominator bounded away from zero in low-variance regions. Wang
et al.
[
36
]
propose the defaults
(
K
1
,
K
2
)
=
(
0.01
,
0.03
)
(K_{1},K_{2})=(0.01,0.03)
, which we retain for the
evaluation metric
, so that the reported SSIM numbers are comparable to prior literature and preserve the discriminating power characterised in
Appendix
D
, but modify for training as a numerical-conditioning knob (see below). The image-level index
SSIM
¯
​
(
Y
^
,
Y
)
\overline{\mathrm{SSIM}}(\hat{Y},Y)
is the mean of
Equation
2
over all windows, and the loss is
ℒ
SSIM
=
1
−
SSIM
¯
​
(
Y
^
,
Y
)
\mathcal{L}_{\mathrm{SSIM}}=1-\overline{\mathrm{SSIM}}(\hat{Y},Y)
.
Multi-scale SSIM
[
35
]
decomposes
Equation
2
as
SSIM
=
l
⋅
c
⋅
s
\mathrm{SSIM}=l\cdot c\cdot s
(luminance, contrast, and structure) and evaluates these components at
M
=
5
M=5
dyadically downsampled resolutions, indexed
j
=
1
j=1
(finest) to
j
=
M
j=M
(coarsest). Contrast
c
j
c_{j}
and structure
s
j
s_{j}
are aggregated across all scales, while luminance is retained only at the coarsest scale,
l
M
l_{M}
, since it is a low-frequency property that does not benefit from multi-scale repetition:
MS
​
-
​
SSIM
​
(
Y
^
,
Y
)
=
[
l
M
​
(
Y
^
,
Y
)
]
α
M
​
∏
j
=
1
M
[
c
j
​
(
Y
^
,
Y
)
]
β
j
​
[
s
j
​
(
Y
^
,
Y
)
]
γ
j
,
\mathrm{MS\text{-}SSIM}(\hat{Y},Y)\;=\;\bigl[l_{M}(\hat{Y},Y)\bigr]^{\alpha_{M}}\prod_{j=1}^{M}\bigl[c_{j}(\hat{Y},Y)\bigr]^{\beta_{j}}\bigl[s_{j}(\hat{Y},Y)\bigr]^{\gamma_{j}},
(3)
with exponents fixed to the calibrated weights of Wang
1
1
1
We follow the standard convention
β
j
=
γ
j
=
w
j
\beta_{j}=\gamma_{j}=w_{j}
for
j
=
1
,
…
,
M
j=1,\dots,M
with
α
M
=
β
M
\alpha_{M}=\beta_{M}
, and
∑
j
w
j
=
1
\sum_{j}w_{j}=1
, using
w
=
(
0.0448
,
0.2856
,
0.3001
,
0.2363
,
0.1333
)
w=(0.0448,0.2856,0.3001,0.2363,0.1333)
.
et al.
[
35
]
. The training loss is
ℒ
MS
​
-
​
SSIM
=
1
−
MS
​
-
​
SSIM
​
(
Y
^
,
Y
)
\mathcal{L}_{\mathrm{MS\text{-}SSIM}}=1-\mathrm{MS\text{-}SSIM}(\hat{Y},Y)
.
Combined losses.
Combined losses pair a structural term with a pixel-wise term, following Zhao
et al.
[
39
]
, to combine the low-frequency fidelity of pixel-wise losses with the local structural sensitivity of SSIM. The two combined-loss families are
ℒ
SSIM
+
L1
​
(
α
)
\displaystyle\mathcal{L}_{\mathrm{SSIM+L1}}(\alpha)
=
α
​
ℒ
SSIM
+
(
1
−
α
)
​
ℒ
L1
,
\displaystyle\;=\;\alpha\,\mathcal{L}_{\mathrm{SSIM}}\;+\;(1-\alpha)\,\mathcal{L}_{\mathrm{L1}},
(4)
ℒ
MS
​
-
​
SSIM
+
L1
​
(
α
)
\displaystyle\mathcal{L}_{\mathrm{MS\text{-}SSIM+L1}}(\alpha)
=
α
​
ℒ
MS
​
-
​
SSIM
+
(
1
−
α
)
​
ℒ
L1
,
\displaystyle\;=\;\alpha\,\mathcal{L}_{\mathrm{MS\text{-}SSIM}}\;+\;(1-\alpha)\,\mathcal{L}_{\mathrm{L1}},
(5)
with
α
∈
{
0.2
,
0.3
,
0.5
,
0.8
}
\alpha\in\{0.2,\,0.3,\,0.5,\,0.8\}
swept across configurations; the primary reported model and the reference for all paired tests is
ℒ
MS
​
-
​
SSIM
+
L1
\mathcal{L}_{\mathrm{MS\text{-}SSIM+L1}}
at
α
=
0.5
\alpha=0.5
.
Numerical conditioning of SSIM-based training losses.
SSIM and MS-SSIM are rational functions whose denominator
(
μ
x
2
+
μ
y
2
+
C
1
)
​
(
σ
x
2
+
σ
y
2
+
C
2
)
(\mu_{x}^{2}+\mu_{y}^{2}+C_{1})(\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2})
can collapse towards its
C
C
-floors in low-texture regions, primarily through the contrast/structure factor
σ
x
2
+
σ
y
2
+
C
2
\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}
in uniform brain parenchyma and large CSF pools, and through both factors in background air (which clips to
0
0
under our normalization). With
L
=
1
L=1
the Wang
et al.
defaults evaluate to
C
1
=
10
−
4
C_{1}=10^{-4}
and
C
2
=
×
10
−
4
C_{2}=9\!\times\!10^{-4}
. On a random sample of 5000 training slices under the brain window, we compute the
×
11
11\!\times\!11
Gaussian-weighted local variance (the same window operator that appears inside SSIM, with
σ
=
1.5
\sigma=1.5
) at every pixel of every slice. Restricted to windows whose local mean intensity exceeds
0.15
0.15
(a threshold that corresponds to approximately
−
1
-1
HU under the
[
−
20
​
HU
,
107
​
HU
]
[-20\,\text{HU},107\,\text{HU}]
brain window, chosen to exclude background air while retaining soft tissue, CSF, parenchyma, and bone), the median local variance is
×
10
−
3
1.0\!\times\!10^{-3}
, numerically indistinguishable from the default
C
2
=
×
10
−
4
C_{2}=9\!\times\!10^{-4}
. In those windows the denominator
σ
x
2
+
σ
y
2
+
C
2
\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}
is dominated by the stability floor rather than by signal, and the resulting near-zero-denominator regime produces large, oscillatory gradients and overflow to NaN during early training. We therefore combine two interventions, each addressing a distinct failure mode of the raw formulation:
1.
Larger
K
2
K_{2}
.
We set
K
1
=
0.01
K_{1}=0.01
and
K
2
=
0.4
K_{2}=0.4
for all SSIM-based training losses, raising
C
2
C_{2}
from
×
10
−
4
9\!\times\!10^{-4}
to
0.16
0.16
(roughly two orders of magnitude). On the same 5000-slice sample,
C
2
=
0.16
C_{2}=0.16
exceeds
99.0
%
99.0\%
of all window variances and
96.5
%
96.5\%
of those above the background-air threshold defined above, so the denominator-stability floor of
Equation
2
dominates the natural within-window variance in all but the highest-contrast edges.
K
1
=
0.01
K_{1}=0.01
is retained because the luminance-term denominator
μ
x
2
+
μ
y
2
+
C
1
\mu_{x}^{2}+\mu_{y}^{2}+C_{1}
is not the limiting factor in our data. The choice is restricted to the training loss; test-set SSIM and MS-SSIM remain reported at the Wang
et al.
defaults for comparability with prior work.
Appendix
D
quantifies the evaluation-metric sensitivity to
K
2
K_{2}
: on the same test predictions, reporting at
K
2
=
0.4
K_{2}=0.4
rather than
0.03
0.03
uniformly inflates mean SSIM by
≈
0.09
\approx 0.09
but compresses the inter-model spread by a factor of
1.57
1.57
, so the Wang
et al.
choice at evaluation is both the literature-comparable and the more discriminating setting.
2.
FP32 autocast exclusion.
SSIM’s variance and covariance terms are second-order statistics computed via the identity
σ
2
=
𝔼
⁡
[
X
2
]
−
(
𝔼
⁡
[
X
]
)
2
\sigma^{2}=\mathbb{E}[X^{2}]-(\mathbb{E}[X])^{2}
, i.e., the mean of the squared pixel values in the window minus the squared window mean. In low-texture windows these two quantities are close, so their difference is the small tail left after many matching leading digits. This subtraction of near-equal numbers is catastrophic in half precision (
float16
, the IEEE 16-bit floating-point format used by default in mixed-precision training): most of the matching digits are lost, and the result can emerge with essentially random sign. Mathematically
σ
2
\sigma^{2}
is non-negative; computed through this identity in
float16
, it can be returned slightly negative. This contaminates two operations in implementations that follow this algebraic form: the
σ
x
2
+
σ
y
2
+
C
2
\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}
denominator of
Equation
2
can flip sign or pass through zero, and the MS-SSIM aggregate
∏
j
(
c
j
​
s
j
)
w
j
\prod_{j}(c_{j}s_{j})^{w_{j}}
raises possibly-negative per-scale factors to fractional exponents
w
j
w_{j}
, yielding NaN; the resulting NaN gradient then propagates through backpropagation and terminates the run. We therefore compute every SSIM and MS-SSIM forward pass in full
float32
precision
2
2
2
The SSIM subgraph runs with the autocast context disabled and its inputs cast to
float32
; the convolutional backbone remains in mixed precision, the PyTorch AMP default that keeps most operators in half precision while automatically promoting numerically sensitive reductions to
float32
.
, while the convolutional backbone remains in mixed precision for throughput.
Interventions 1–2 eliminated the most common failure mode (within-epoch NaN at initialization) but did not eliminate training-time divergence (
Appendix
B
) entirely: a substantial fraction of SSIM-family configurations at smaller batch sizes and at particular learning rates still diverged. The full divergence structure and its dependence on learning rate and batch size are analyzed in
Section
5.2
, and the per-family, per-batch-size pass/fail accounting is given in Appendix
B
.
Hyperparameter sweep.
Each loss family was trained across learning rates in the range
[
10
−
4
,
×
10
−
3
]
[10^{-4},\,3\!\times\!10^{-3}]
at the default batch size of 96. SSIM-family configurations additionally include smaller batch sizes where indicated, and three non-SSIM runs (two L1, one MSE) are repeated at batch size 64 as a stability check; the per-family, per-batch-size coverage is tabulated in
Table
8
. The mixing weight
α
∈
{
0.2
,
0.3
,
0.5
,
0.8
}
\alpha\in\{0.2,0.3,0.5,0.8\}
is varied in combined-loss configurations as in Equations (
4
)–(
5
). Five MS-SSIM+L1 recovery-attempt fine-tunes, additional to this systematic sweep, start from diverged MS-SSIM+L1 checkpoints at learning rate (lr)
∈
{
×
10
−
5
,
10
−
4
,
×
10
−
4
}
\in\{5\!\times\!10^{-5},\,10^{-4},\,3\!\times\!10^{-4}\}
, batch size (bs)
=
64
=64
, and a 300-epoch budget (constant LR for three of the five); all five diverged and are included in the divergence accounting of Appendix
B
but do not enter any reported metric. All remaining hyperparameters are fixed as described in
Section
3.5
.
3.5
Optimization and training protocol
All experiments use AdamW
[
25
]
with default
(
β
1
,
β
2
,
ε
)
(\beta_{1},\beta_{2},\varepsilon)
and weight decay
0.01
0.01
. The learning rate follows a linear warmup over the first 5 epochs from
0
0
to the configured peak and then cosine-anneals to
η
min
=
10
−
6
\eta_{\min}=10^{-6}
over the remaining 495 epochs of the 500-epoch budget; gradients are clipped at
∥
⋅
∥
2
≤
1
\lVert\cdot\rVert_{2}\leq 1
. Training is mixed-precision, with the SSIM and MS-SSIM forward passes kept in
float32
as described in intervention 2 of
Section
3.4
. Early stopping monitors the validation loss with patience
15
15
and minimum delta
10
−
4
10^{-4}
; the monitored quantity is loss-family-specific by construction, so each run is halted against its own validation trajectory rather than a shared surrogate. The train/validation split is patient-wise (no slices from a validation patient appear in training).
Table
2
lists the shared hyperparameters; environment, hardware, seeding, and determinism settings are reported in
Appendix
A
.
Table 2
:
Shared training hyperparameters across all experiments.
Parameter
Value
Optimizer
AdamW,
(
β
1
,
β
2
,
ε
)
=
(
0.9
,
0.999
,
10
−
8
)
(\beta_{1},\beta_{2},\varepsilon)=(0.9,0.999,10^{-8})
, weight decay
=
0.01
=0.01
Scheduler
Linear warmup (5 epochs,
0
→
peak
0\to\text{peak}
), then cosine to
η
min
=
10
−
6
\eta_{\min}=10^{-6}
Gradient clipping
∥
⋅
∥
2
≤
1
\lVert\cdot\rVert_{2}\leq 1
Max epochs
500 (early stopping triggers first)
Early stopping
monitor: validation loss (min); patience
=
15
=15
; min. delta
=
10
−
4
=10^{-4}
Default batch size
96 (varied: 32, 64, 96)
SSIM window
11
×
11
11\times 11
(Wang
et al.
default)
MS-SSIM scales
5 (Wang
et al.
default)
Train / validation split
80% / 20% of non-test patients, patient-wise
Test set
30 patients (subtype-stratified selection)
Hardware
1
×
\times
NVIDIA GeForce RTX 3080 Ti (12 GB)
3.6
Evaluation metrics
We report five complementary metrics: the structural similarity index (SSIM) and its multi-scale variant (MS-SSIM), mean absolute error (MAE), gradient MAE, and peak signal-to-noise ratio (PSNR). Higher is better for SSIM, MS-SSIM, and PSNR; lower is better for MAE and gradient MAE. All metrics are computed on the reconstructed
512
×
512
512\times 512
prediction and target slices (
Figure
2
(c)) in the same
[
0
,
1
]
[0,1]
-normalized representation used for training, i.e. after the center-
44
44
HU / width-
128
128
HU window of
Section
3.1
; no further windowing or rescaling is applied at evaluation time. The concrete definitions are as follows.
•
SSIM, MS-SSIM.
As defined by Equations (
2
)–(
3
) with
11
×
11
11\times 11
Gaussian windows,
M
=
5
M=5
scales, the calibrated weights of Wang
et al.
[
35
]
,
w
=
(
0.0448
,
0.2856
,
0.3001
,
0.2363
,
0.1333
)
w=(0.0448,0.2856,0.3001,0.2363,0.1333)
, and the Wang
et al.
[
36
]
defaults
(
K
1
,
K
2
)
=
(
0.01
,
0.03
)
(K_{1},K_{2})=(0.01,\,0.03)
restored for evaluation, in contrast to the training-only choice
K
2
=
0.4
K_{2}=0.4
motivated in intervention 1 of
Section
3.4
.
•
MAE.
Pixel-wise mean absolute error,
1
N
​
∑
i
|
y
^
i
−
y
i
|
\tfrac{1}{N}\sum_{i}\lvert\hat{y}_{i}-y_{i}\rvert
; identical in definition to
ℒ
L1
\mathcal{L}_{\mathrm{L1}}
of
Equation
1
, reported here as a metric on the normalized intensities.
•
Gradient MAE.
Edge-preservation metric:
3
×
3
3\times 3
Sobel operators are applied to
Y
^
\hat{Y}
and
Y
Y
separately in the horizontal and vertical directions, gradient magnitudes
∥
∇
I
∥
=
G
x
2
+
G
y
2
\lVert\nabla I\rVert=\sqrt{G_{x}^{2}+G_{y}^{2}}
are computed for each, and the metric is the mean absolute error between the two magnitude maps.
•
PSNR.
PSNR
⁡
(
Y
^
,
Y
)
=
10
​
log
10
⁡
(
L
2
/
MSE
⁡
(
Y
^
,
Y
)
)
\mathrm{PSNR}(\hat{Y},Y)=10\log_{10}\!\bigl(L^{2}/\mathrm{MSE}(\hat{Y},Y)\bigr)
in decibels, with peak intensity
L
=
1
L=1
; pairs with
MSE
=
0
\mathrm{MSE}=0
are assigned
+
∞
+\infty
and excluded from averages.
Metrics are evaluated on every interpolated slice of every test triplet; slice-level values are averaged within each patient to form the patient-level quantities used in the statistical analysis of
Section
3.8
. Because the reference middle slice
I
k
+
1
I_{k+1}
carries acquisition noise, a model that perfectly recovers the conditional-expectation signal of
Section
5.4
would differ from it by the suppressed noise; see
Section
5.7
for the implications when comparing top-performing losses.
3.7
ROI-level noise quantification
We quantify the implicit-denoising property analysed in
Section
5.4
via the ROI-level protocol below.
To quantify within-tissue noise in held-out test predictions, we measure the pixel standard deviation in three anatomically fixed white-matter regions of interest (ROIs) per slice: left and right centrum semiovale, and pons. Each ROI is a
16
×
16
16\times 16
-pixel patch placed by a brain-mask-based heuristic (centroid and bounding box of the largest connected component at threshold
0.05
0.05
); ROIs that extend outside the brain mask, overlap a hemorrhage-labelled slice, or contain intensity outliers exceeding three times the patient median ROI standard deviation are rejected. Of
968
×
3
=
2904
968\times 3=2904
candidate ROI–triplet combinations,
1262
1262
are retained after rejection (
396
396
dropped for hemorrhage overlap,
183
183
for geometry,
219
219
for outliers,
18
18
pons ROIs unplaceable). Within each surviving ROI we compute the pixel standard deviation
σ
ROI
\sigma_{\mathrm{ROI}}
of the acquired middle slice and of each model’s prediction, and define the noise-reduction ratio
η
=
(
σ
acq
−
σ
pred
)
/
σ
acq
\eta=(\sigma^{\mathrm{acq}}-\sigma^{\mathrm{pred}})/\sigma^{\mathrm{acq}}
(positive: the prediction has lower in-ROI variance than the acquired reference). Per-patient
η
\eta
is the mean over surviving ROIs within each triplet and then over triplets within each patient; we test each model against zero with a paired Wilcoxon signed-rank across the
28
28
patients with any surviving ROI, and apply Benjamini–Hochberg FDR across the 5-model family. The radially-averaged noise power spectrum (NPS) of the same ROIs, computed following ICRU 54 conventions, complements the scalar analysis by showing which spatial frequencies are suppressed.
3.8
Statistical analysis
Primary inference is
patient-level
(30 independent patients). The paired metric differences are heavy-tailed, so we use nonparametric procedures throughout:
1.
compute metric means per patient;
2.
estimate mean and 95% CI via nonparametric percentile bootstrap (10,000 resamples)
[
8
]
;
3.
perform two-sided paired Wilcoxon signed-rank tests
[
37
]
of each candidate against the reference model (MS-SSIM+L1, 0.5/0.5), a one-vs-reference design rather than an all-pairs omnibus, with the reference designated
a priori
(rationale and candidate list in
Section
4.2
).
Patient-level pairing ensures that each model comparison tests whether one loss consistently outperforms the other across the same 30 anatomies, thereby controlling for inter-patient variance rather than relying on absolute metric magnitudes. We adopt a significance level of
α
=
0.05
\alpha=0.05
for all tests.
To control the false discovery rate across the patient-level paired-test family (50 tests: 10 candidate models
×
\times
5 reported metrics; full enumeration in
Section
4.2
), we apply the Benjamini–Hochberg procedure
[
1
]
at
q
<
0.05
q<0.05
; BH-adjusted
q
q
-values are reported alongside raw
p
p
-values. Slice-level analyses are also computed for transparency, but we avoid treating slice-level
p
p
-values as primary due to within-patient dependence.
Exploratory subgroup analysis.
Separate from the primary patient-level paired comparisons,
Section
4.3
reports an
exploratory, post-hoc
subgroup analysis testing whether slice-level metrics differ between hemorrhage and normal target slices. Because the two subgroups are different slices (not paired), we use the two-sample Mann–Whitney
U
U
test (two-sided), with Benjamini–Hochberg FDR applied jointly across the
8
×
3
8\times 3
primary-metric hemorrhage-analysis family and a patient-level cluster bootstrap reported as an independence-aware sensitivity check; effect-size and bootstrap definitions are given in
Appendix
C
.
Resolution limits.
No a priori power calculation was performed; reported effect sizes are observed rather than targeted. The two-sided exact Wilcoxon signed-rank test at
n
=
30
n=30
has a minimum attainable
p
p
-value of
1.86
×
10
−
9
1.86\times 10^{-9}
, which bounds patient-level significance claims from below. Slice-level tests are reported for transparency but can be anti-conservative under within-patient dependence; the hemorrhage-stratification analysis therefore reports patient-level cluster-bootstrap
p
p
-values as an independence-aware sensitivity check.
