# Blind Ultrasound Image Enhancement via Self-Supervised Physics-Guided Degradation Modeling

paper_id: arxiv:2601.21856v1
tier: H
source_used: html_arxiv
warning: none

## Intro

\IEEEPARstart
Ultrasound (US) imaging is one of the most widely used diagnostic tools due to its real-time capability, safety, and low cost. Nevertheless, the diagnostic value of ultrasound is often limited by speckle noise and blurring caused by the system’s point spread function (PSF). These degradations obscure subtle anatomical features and degrade the performance of downstream tasks such as segmentation and classification. Conventional image enhancement methods require explicit knowledge of noise variance or PSF characteristics, which are rarely available in practice. Recent deep learning–based approaches have achieved significant progress, but most rely on supervised training with clean ground-truth images—an unrealistic assumption in clinical settings.
Classical approaches include anisotropic diffusion, wavelet shrinkage, and non-local means (NLM). Variants such as speckle reducing anisotropic diffusion (SRAD)
[
29
]
, fractional-order diffusion
[
1
]
, and adaptive nonlinear diffusion
[
15
]
improve speckle suppression while preserving edges. Wavelet-based thresholding methods
[
4
,
18
,
23
]
exploit multiscale representations for denoising, whereas NLM
[
3
]
and its extensions
[
9
,
21
,
34
,
16
]
leverage patch similarity for noise reduction. Low-rank and sparse priors have also been explored, such as high-order SVD
[
24
]
and non-convex optimization strategies
[
31
]
.
Deep learning has enabled end-to-end ultrasound enhancement frameworks. CNN-based models such as DnCNN
[
33
]
and UNet
[
25
]
are commonly adapted for medical image restoration, while autoencoder-based methods
[
6
]
and residual dense block architectures
[
22
]
aim to preserve diagnostically critical details. Self-supervised techniques including Noise2Void
[
17
]
and Noise2Self
[
2
]
have demonstrated that models can be trained directly on corrupted images without clean references, although their application to ultrasound remains limited. Another key challenge is resolution loss, which is compounded by denoising. Prior work has explored adaptive beamforming
[
13
,
10
,
11
]
, deconvolution
[
12
,
12
]
, and physics-guided deep learning
[
14
]
, however the methods are either assumption-heavy or require access to channel data which is not available in clinical settings.
Despite these advances, three challenges remain: (i) reliance on clean ground-truth data, (ii) limited robustness to diverse degradations, and (iii) resolution loss during denoising. To address these issues, we propose a blind, self-supervised framework for ultrasound enhancement. The method leverages a Swin Convolutional UNet (SC-UNET)
[
32
]
trained on synthetic input–target pairs derived from real ultrasound data using physics-guided degradation modeling. The augmentation pipeline introduces diverse degradations, including Fourier-domain perturbations, Gaussian and multiplicative speckle noise, and random PSF blurs, applied to both natural images and different modalities of actual ultrasound scans. For natural images, the original non-degraded images serve as ground truth, whereas for ultrasound scans, clean-like targets are approximated using non-local low-rank (NLLR) denoising
[
35
]
. This strategy eliminates the need for manually curated ground truth while exposing the model to realistic noise conditions. Furthermore, the hierarchical Swin-based encoder–decoder structure enhances both local and global feature modeling, resulting in improved speckle suppression, boundary preservation, and robustness across heterogeneous datasets.
Furthermore, recent efforts in cross-device harmonization
[
26
]
emphasize the importance of generalization across diverse ultrasound scanners. Our framework complements such efforts by providing a plug-and-play module that enhances image quality consistently, thereby improving visualization and the reliability of downstream analysis in clinical workflows.

## Method

Figure 1:
Physics-guided noise degradation pipeline used for training data augmentation.
2.1
Overview
The proposed framework is a Swin convolution-based encoder–decoder network designed to enhance grayscale ultrasound images. Training pairs are synthetically generated from real ultrasound scans using a physics-guided degradation model (shown in Fig.
1
). This model introduces additive Gaussian noise, Fourier-domain perturbations, and random blurring (to simulate PSF effects). For clean-like targets, non-local low-rank (NLLR)
[
35
]
denoising is applied, enabling self-supervised learning without requiring ground-truth clean images.
2.2
Physics–Guided Degradation Model
Let
𝐈
∈
[
0
,
1
]
H
×
W
\mathbf{I}\in[0,1]^{H\times W}
denote a grayscale ultrasound frame. During training we form patches by
(a) resizing to
128
×
128
128{\times}128
, (b) applying a random in–plane rotation
θ
∼
𝒰
⁡
[
−
15
∘
,
15
∘
]
\theta\!\sim\!\mathcal{U}[-15^{\circ},15^{\circ}]
, and
(c) taking a random
64
×
64
64{\times}64
crop of the rotated image. Denote this augmented patch by
𝐈
~
\tilde{\mathbf{I}}
.
We then synthesize degraded inputs via two families of operators that reflect common ultrasound artifacts:
(1) Blur by a PSF surrogate.
We convolve
𝐈
~
\tilde{\mathbf{I}}
with an isotropic Gaussian kernel of odd size
k
∈
{
3
,
5
,
7
,
9
,
11
,
13
,
15
,
17
}
k\!\in\!\{3,5,7,9,11,13,15,17\}
:
𝐈
b
=
𝐈
~
∗
h
k
,
h
k
​
(
u
,
v
)
∝
exp
⁡
(
−
u
2
+
v
2
2
​
σ
b
2
)
,
\mathbf{I}_{b}=\tilde{\mathbf{I}}*h_{k},\qquad h_{k}(u,v)\propto\exp\!\left(-\frac{u^{2}+v^{2}}{2\sigma_{b}^{2}}\right),
(1)
with
σ
b
\sigma_{b}
chosen so that
k
≈
2
​
⌈
3
​
σ
b
⌉
+
1
k\approx 2\lceil 3\sigma_{b}\rceil+1
. This approximates acquisition blur from the system PSF.
(2) Noise processes.
We inject two complementary noise types:
•
Spatial additive Gaussian noise
(to emulate thermal/receiver noise):
𝐈
g
=
𝐈
~
+
𝐧
,
𝐧
​
∼
i.i.d.
​
𝒩
​
(
0
,
σ
g
2
)
,
σ
g
∼
𝒰
⁡
(
0.05
,
0.20
)
,
\mathbf{I}_{g}=\tilde{\mathbf{I}}+\mathbf{n},\qquad\mathbf{n}\overset{\text{i.i.d.}}{\sim}\mathcal{N}(0,\sigma_{g}^{2}),\;\;\sigma_{g}\sim\mathcal{U}(0.05,0.20),
(2)
followed by clipping to
[
0
,
1
]
[0,1]
.
•
Fourier–domain complex perturbations
(to mimic phase/magnitude distortions and speckle–like granular texture):
ℱ
⁡
(
𝐈
f
)
=
(
1
−
γ
f
)
​
ℱ
​
(
𝐈
~
)
+
γ
f
​
‖
ℱ
⁡
(
𝐈
~
)
‖
∞
​
𝜻
,
𝜻
∼
𝒞
​
𝒩
​
(
0
,
𝐈
)
,
\mathcal{F}(\mathbf{I}_{f})=(1-\gamma_{f})\,\mathcal{F}(\tilde{\mathbf{I}})+\gamma_{f}\,\|\mathcal{F}(\tilde{\mathbf{I}})\|_{\infty}\,\boldsymbol{\zeta},\quad\boldsymbol{\zeta}\sim\mathcal{CN}(0,\mathbf{I}),
(3)
where
𝒞
​
𝒩
\mathcal{CN}
denotes zero–mean complex Gaussian noise (independent real/imaginary parts),
γ
f
∼
𝒰
⁡
(
0
,
0.2
)
\gamma_{f}\!\sim\!\mathcal{U}(0,0.2)
in training, and
𝐈
f
=
|
ℱ
−
1
​
(
ℱ
⁡
(
𝐈
f
)
)
|
\mathbf{I}_{f}=\big|\mathcal{F}^{-1}(\mathcal{F}(\mathbf{I}_{f}))\big|
.
(3) Stochastic composition (order matters).
Rather than linearly mixing degradations, we
compose
them in random order to avoid biasing the network to a single corruption sequence:
𝐈
d
=
𝒯
⁡
(
𝐈
~
)
,
𝒯
∈
{
𝒩
∘
ℬ
⏟
blur
→
noise
,
ℬ
∘
𝒩
⏟
noise
→
blur
}
,
\mathbf{I}_{d}=\mathcal{T}(\tilde{\mathbf{I}}),\qquad\mathcal{T}\in\Big\{\underbrace{\mathcal{N}\!\circ\!\mathcal{B}}_{\text{blur $\to$ noise}},\underbrace{\mathcal{B}\!\circ\!\mathcal{N}}_{\text{noise $\to$ blur}}\Big\},
(4)
where
ℬ
⁡
(
⋅
)
\mathcal{B}(\cdot)
is Gaussian blur with
k
∈
{
3
,
5
,
…
,
17
}
k\in\{3,5,\ldots,17\}
, and
𝒩
⁡
(
⋅
)
\mathcal{N}(\cdot)
is sampled uniformly from the two noise families above.
Concretely, with probability
>
0.55
>0.55
we apply
blur
→
\to
noise
using a random
k
k
,
and independently with probability
<
0.45
<0.45
we apply a light
noise
→
\to
blur
path
(
Fourier noise
followed by
k
=
3
k{=}3
), so both may occur, yielding a diverse corruption space.
At test time, controlled stress tests are created by setting a user–specified
Fourier noise strength (
γ
f
∈
[
0
,
1
]
\gamma_{f}\!\in[0,1]
) and/or a fixed blur size
k
k
.
(4) Targets and normalization.
All intensities are kept in
[
0
,
1
]
[0,1]
with clipping after each corruption step.
For natural images, the original clean image serves as the target.
For ultrasound scans, clean–like targets are approximated via non–local low–rank denoising:
𝐈
t
=
𝒟
NLLR
​
(
𝐈
)
,
\mathbf{I}_{t}=\mathcal{D}_{\text{NLLR}}(\mathbf{I}),
(5)
removing the need for ground–truth clean acquisitions.
Discussion.
The Gaussian PSF surrogate broadens edges in a controllable manner, while the k–space perturbations inject phase/magnitude fluctuations that manifest as fine–grain texture after inverse FFT, closely resembling the granular statistics observed in ultrasound B–mode. Alternating the operator order (blur
→
\to
noise vs. noise
→
\to
blur) intentionally changes edge–to–noise interactions, preventing the model from overfitting to a single artifact chronology. In practice we find this composition strategy yields superior robustness to unknown scanner settings and consistently improves both fidelity (PSNR) and structure (SSIM) in downstream evaluations.
2.3
Architecture
Figure 2:
Hybrid Swin-Convolution block. Input channels are split evenly into a local conv path and a shifted–window attention path, fused with
1
×
1
1{\times}1
projection, and added residually to the input.
Figure 3:
Overall Swin–Conv U–Net. Three encoder stages (
↓
\downarrow
), bottleneck, and three decoder stages (
↑
\uparrow
) with additive cross–scale fusions. Downsampling uses stride–2 conv; upsampling uses stride–2 transposed conv.
Our backbone is a Swin–Transformer
[
19
]
augmented U–Net (“Swin–Conv U–Net”), closely following the practical SC-UNet design
[
32
]
. It couples local convolutions (robust to speckle statistics) with shifted–window self–attention to capture mid/long range context, and uses additive cross–scale fusions rather than concatenation. Figure
2
depicts the core hybrid block; Figure
3
shows the overall topology.
Implementation used in this work.
Unless otherwise stated, we use
SCUNet(in_nc=1, dim=64, config=[2,2,2,2,2,2,2])
, window size
M
=
8
M{=}8
, head dimension
d
h
=
32
d_{h}{=}32
, and a linear drop–path schedule set to
0
0
(disabled). Replication padding brings inputs to the nearest multiple of
64
64
on each side; outputs are cropped back to the original size.
Patch embedding and stem.
A
3
×
3
3{\times}3
conv maps the grayscale input
𝐈
∈
[
0
,
1
]
H
×
W
×
1
\mathbf{I}\!\in\![0,1]^{H\times W\times 1}
to
C
=
64
C{=}64
channels (“stem” features). This improves edge fidelity over a purely linear embedding and provides a stable spatial bias for speckle.
Encoder–bottleneck–decoder hierarchy.
We employ three downsampling stages, a bottleneck, and three symmetric upsampling stages:
•
Down path:
Each stage consists of
N
s
N_{s}
hybrid blocks (see below) followed by a stride–2
2
×
2
2{\times}2
convolution that halves the spatial size and doubles channels:
→
→
→
512
64\!\to\!128\!\to\!256\!\to\!512
.
•
Bottleneck:
Hybrid blocks operate at the coarsest scale (512 channels) where global context is cheapest.
•
Up path:
Each stage starts with a stride–2 transposed convolution (e.g.,
→
256
512\!\to\!256
), then hybrid blocks refine features. Instead of U–Net concatenation, we use
additive
fusion with the encoder features at the same scale (
x
+
x
enc
x\!+\!x_{\text{enc}}
); this keeps channel counts fixed and reduces memory without hurting detail.
•
Reconstruction head:
A
3
×
3
3{\times}3
conv produces the
1
1
–channel output. We add a final residual with the stem features (
x
+
x
stem
x\!+\!x_{\text{stem}}
) to preserve fine detail and stabilize training.
Hybrid Swin–Convolution block (
TransConvBlock
).
Given an input tensor with
C
C
channels at some resolution, we first apply a
1
×
1
1{\times}1
conv and split channels evenly into a
conv branch
(
C
/
2
C/2
) and a
transformer branch
(
C
/
2
C/2
), process them in parallel, concatenate, project with another
1
×
1
1{\times}1
conv, and add a residual from the block input:
𝐘
=
𝐗
+
ϕ
1
×
1
​
(
[
f
conv
​
(
𝐗
c
)
⏟
local
,
f
swin
​
(
𝐗
t
)
⏟
nonlocal
]
)
⏟
fusion
.
\mathbf{Y}\;=\;\mathbf{X}\;+\;\underbrace{\phi_{1\times 1}\!\Big([\;\underbrace{f_{\text{conv}}(\mathbf{X}_{c})}_{\text{local}}\;,\;\underbrace{f_{\text{swin}}(\mathbf{X}_{t})}_{\text{nonlocal}}\;]\Big)}_{\text{fusion}}.
The conv branch is a lightweight
3
×
3
3{\times}3
– ReLU –
3
×
3
3{\times}3
residual unit. The transformer branch is a
Swin block
(described below) operating on
(
H
×
W
)
(H{\times}W)
tokens with channel width
C
/
2
C/2
.
Swin block with relative position bias.
Each block applies LayerNorm and windowed multi–head self–attention (W–MSA) with window size
M
=
8
M=8
and learnable 2D relative positional bias, followed by an MLP with GELU
[
8
]
; both subpaths are wrapped with residual connections:
𝐙
=
𝐗
+
W
​
-
​
MSA
​
(
LN
⁡
(
𝐗
)
)
,
𝐘
=
𝐙
+
MLP
⁡
(
LN
⁡
(
𝐙
)
)
.
\mathbf{Z}=\mathbf{X}+\mathrm{W\text{-}MSA}(\mathrm{LN}(\mathbf{X})),\quad\mathbf{Y}=\mathbf{Z}+\mathrm{MLP}(\mathrm{LN}(\mathbf{Z})).
We alternate plain and
shifted
windows (
W
/
SW
) across consecutive blocks to enable cross–window interaction (
SW
is automatically disabled when the current resolution
≤
M
\leq M
). The number of heads is determined by the transformer–branch width:
h
=
C
/
2
d
h
h=\tfrac{C/2}{d_{h}}
(e.g.,
h
=
1
,
2
,
4
,
8
h=1,2,4,8
at channel widths
32
,
64
,
128
,
256
32,64,128,256
).
2.4
Datasets
Figure 4:
Sample images from datasets: (a) UDIAT, (b) JNU-IFM, (c) XPIE Set-P, and (d) PSFHS Test.
We evaluate the proposed method using three public datasets for training/validation and one benchmark dataset for downstream segmentation. Unless noted, all images are used in grayscale B-mode.
•
UDIAT Dataset B
[
28
]
: Breast ultrasound dataset with tumor annotations, collected at the UDIAT Diagnostic Center, Spain, using a Siemens ACUSON scanner. The average image size is
760
×
570
760\times 570
pixels. Publicly available at
https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php
. Samples used: 131 (train), 16 (val), 16 (test).
•
JNU-IFM
[
20
]
: Intrapartum fetal monitoring dataset with grayscale 2D transperineal ultrasound scans. It contains 6224 images extracted from 78 videos of 51 patients, acquired using a Youkey D8 wireless probe. Labels were validated by expert radiologists. Available at
https://figshare.com/articles/dataset/JNU-IFM/14371652
. Samples used: 4224 (train), 1000 (val), 1000 (test).
•
XPIE Set-P
[
27
]
: Natural images with ground-truth masks used to expose the model to cross-domain textures and to synthesize phantom-like inputs during degradation modeling. Available at
http://cvteam.net/projects/CVPR17-ELE/XPIE.tar.gz
.
Split:
850 (train), 200 (val), 200 (test).
•
PSFHS
[
5
]
: Intrapartum transperineal ultrasound annotated for pubic symphysis and fetal head; image size
256
×
256
256\times 256
pixels. Access upon request via
https://ps-fh-aop-2023.grand-challenge.org/
.
Split used in this work:
3200 (train), 800 (val), 700 (test).
Important:
The PSFHS
train/val
splits are used
only
to train the downstream UNet segmentation model; the proposed denoising model is
not
trained on PSFHS. The 700-image test split is held out exclusively for reporting segmentation with/without our preprocessing.
Sample images from each dataset is shown in Figure
4
.
2.5
Quantitative Metrics
We report both
resolution-recovery
and
perceptual/fidelity
metrics. Resolution metrics are computed from 1D intensity profiles extracted from regions of interest (ROIs), while quality metrics are evaluated on full images.
1
1
1
Rule-of-thumb bands below are indicative and depend on sampling, anatomy, and acquisition. When in doubt, compare
relative
improvements to the input or a non-degraded reference.
2.5.1
Resolution Metrics
•
Full Width at Half Maximum (FWHM):
For a 1D profile
p
⁡
(
x
)
p(x)
, the FWHM is the width at half the maximum:
FWHM
=
x
2
−
x
1
,
p
⁡
(
x
1
)
=
p
⁡
(
x
2
)
=
1
2
​
max
x
⁡
p
⁡
(
x
)
.
\mathrm{FWHM}=x_{2}-x_{1},\quad p(x_{1})=p(x_{2})=\tfrac{1}{2}\max_{x}p(x).
(6)
Lower values indicate sharper transitions.
•
Mean Gradient (GradMean):
GradMean
=
1
N
​
∑
i
=
1
N
|
∇
p
​
(
x
i
)
|
.
\mathrm{GradMean}=\frac{1}{N}\sum_{i=1}^{N}\bigl|\nabla p(x_{i})\bigr|.
(7)
Higher is sharper on average.
Rule-of-thumb (images scaled to
[
0
,
1
]
[0,1]
):
ideal
≥
0.05
\geq 0.05
,
good
−
0.05
0.03\!-\!0.05
,
poor
≤
0.02
\leq 0.02
.
•
Maximum Gradient (GradMax):
GradMax
=
max
i
⁡
|
∇
p
​
(
x
i
)
|
.
\mathrm{GradMax}=\max_{i}\bigl|\nabla p(x_{i})\bigr|.
(8)
Captures the steepest edge.
Rule-of-thumb (
[
0
,
1
]
[0,1]
scale):
ideal
≥
0.30
\geq 0.30
,
good
−
0.30
0.18\!-\!0.30
,
poor
≤
0.12
\leq 0.12
.
•
Contrast:
Contrast
=
I
max
−
I
min
I
max
+
I
min
,
\mathrm{Contrast}=\frac{I_{\max}-I_{\min}}{I_{\max}+I_{\min}},
(9)
with
I
max
,
I
min
I_{\max},I_{\min}
the extremal intensities in the ROI; higher is better.
Rule-of-thumb:
ideal
≥
0.95
\geq 0.95
,
good
−
0.95
0.90\!-\!0.95
,
poor
≤
0.80
\leq 0.80
.
2.5.2
Quality Metrics
•
Peak Signal-to-Noise Ratio (PSNR):
PSNR
=
10
​
log
10
​
(
L
2
MSE
)
,
\mathrm{PSNR}=10\log_{10}\left(\frac{L^{2}}{\mathrm{MSE}}\right),
(10)
where
L
L
is the dynamic range and
MSE
\mathrm{MSE}
is the mean squared error between the reference and the reconstruction.
In all our experiments we use
L
=
255
L{=}255
(8-bit range).
2
2
2
If images are normalized to
[
0
,
1
]
[0,1]
, then
L
=
1
L{=}1
. We explicitly rescale to the 8-bit range for consistent reporting.
•
Structural Similarity Index (SSIM):
For patches
x
x
and
y
y
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
\mathrm{SSIM}(x,y)=\frac{(2\mu_{x}\mu_{y}+C_{1})(2\sigma_{xy}+C_{2})}{(\mu_{x}^{2}+\mu_{y}^{2}+C_{1})(\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2})},
(11)
where
μ
x
,
μ
y
\mu_{x},\mu_{y}
are means,
σ
x
2
,
σ
y
2
\sigma_{x}^{2},\sigma_{y}^{2}
variances, and
σ
x
​
y
\sigma_{xy}
the covariance.
In the standard Python implementations, the stabilizers are set via
C
1
=
(
k
1
​
L
)
2
,
C
2
=
(
k
2
​
L
)
2
,
C_{1}=(k_{1}L)^{2},\qquad C_{2}=(k_{2}L)^{2},
with
k
1
=
0.01
k_{1}{=}0.01
,
k
2
=
0.03
k_{2}{=}0.03
and the same
L
=
255
L{=}255
used for PSNR.
2.6
Training and Evaluation Protocol
We train the network to map a degraded input
𝐈
d
\mathbf{I}_{d}
to an enhanced image
𝐈
^
=
f
θ
​
(
𝐈
d
)
\hat{\mathbf{I}}=f_{\theta}(\mathbf{I}_{d})
using paired targets
𝐈
t
\mathbf{I}_{t}
(natural-image originals or NLLR-filtered ultrasound). The objective is a pixelwise
ℓ
1
\ell_{1}
loss:
ℒ
⁡
(
θ
)
=
‖
f
θ
​
(
𝐈
d
)
−
𝐈
t
‖
1
.
\mathcal{L}(\theta)\;=\;\bigl\|\,f_{\theta}(\mathbf{I}_{d})-\mathbf{I}_{t}\,\bigr\|_{1}.
(12)
Hyperparameters and implementation.
We use
num_epochs
=
4000
=4000
,
learning_rate
=
10
−
4
=10^{-4}
, and
batch_size
=
16
=16
. Mini-batches are shuffled every epoch. To accommodate the encoder–decoder strides, inputs are padded along height and width to the nearest multiple of
64
64
and cropped back after inference (no tiling). Degraded inputs
𝐈
d
\mathbf{I}_{d}
are generated on-the-fly via the physics-guided pipeline (random PSF blur, additive Gaussian noise, and Fourier-domain perturbations), etc.
Training data and targets.
The training set contains
5205
images in total: UDIAT B (
131
breast ultrasound images), JNU-IFM (
4224
intrapartum transperineal ultrasound frames), and
850
natural images (with masks) from XPIE Set-P used to synthesize phantom-like textures. For natural images, the clean original serves as
𝐈
t
\mathbf{I}_{t}
; for ultrasound images, clean-like targets are obtained using non-local low-rank denoising
[
35
]
, i.e.,
𝐈
t
=
𝒟
NLLR
​
(
𝐈
)
\mathbf{I}_{t}=\mathcal{D}_{\text{NLLR}}(\mathbf{I})
.
Model selection and validation.
Because ground-truth clean B-mode is unavailable, checkpoint selection combines (i) PSNR/SSIM on validation pairs (natural images and NLLR targets) with (ii) visual inspection on ultrasound validation images (edge sharpness, speckle suppression, artifact avoidance). This composite criterion emphasizes clinical plausibility over any single surrogate metric.
Segmentation-based evaluation.
To quantify downstream utility, we train a UNet
[
25
]
on the PSFHS
training/validation
split (fetal head and pubic symphysis) and evaluate it
with vs. without
the proposed preprocessing. PSFHS images are
not
used to train the denoiser. We report Dice on the independent
700
-image PSFHS
test
set alongside the image-quality metrics from Section
2.5
, thereby assessing domain generalization and the impact on anatomical delineation.
