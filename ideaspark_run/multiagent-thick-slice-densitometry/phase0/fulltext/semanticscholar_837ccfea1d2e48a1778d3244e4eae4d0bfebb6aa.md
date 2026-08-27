# Integrated Forward-Inverse Network for Lensless Image Reconstruction

paper_id: semanticscholar:837ccfea1d2e48a1778d3244e4eae4d0bfebb6aa
tier: T2
source_used: html_arxiv
warning: none

## Intro

Modern optical imaging systems, ranging from compact lensless cameras with coded apertures to advanced microscopes with engineered point-spread functions (PSFs), are increasingly designed with complex forward models. A growing class of such systems operates with PSFs that are intentionally or unavoidably broadened by designed optical coding. In this regime, each measurement mixes scene information over a wide spatial extent, making the inverse problem severely ill-conditioned. While such designs unlock diverse imaging capabilities that transcend the limits of conventional optics
[
40
,
41
,
1
,
2
,
5
]
, they also introduce substantial challenges for reconstruction. In practice, the effective PSFs can vary across both the field of view and channels, violating the stationarity assumptions underlying standard inverse pipelines
[
42
,
49
,
13
]
. Hardware imperfections and residual modulations further deteriorate this model mismatch, making accurate and robust reconstruction a central challenge as optical platforms continue to shrink and diversify.
A wide spectrum of approaches has been explored for image reconstruction in lensless imaging systems. Classical inverse mappings
[
45
]
and model-based optimization methods
[
38
,
28
,
12
]
built on well-defined priors offer physically grounded results, but they are often computationally expensive, sensitive to calibration errors, and unreliable under model mismatch. With advances in deep learning, data-driven methods
[
4
,
35
]
have enabled end-to-end mappings from measurements to target scenes, yet they may not explicitly encode the underlying system physics, which can reduce accuracy and robustness under out-of-distribution conditions; such models may also produce hallucinations. In response, hybrid methods
[
34
,
49
,
24
,
26
]
that embed the physical forward model within a learning framework have emerged, improving efficiency and grounding predictions in the physical model while leveraging data-driven components to capture priors that are difficult to specify analytically.
However, many existing hybrid pipelines incorporate physics in a one-sided manner, either operating primarily in the measurement domain or refining only an already inverted estimate; once an image is reconstructed, the measurement cues become less accessible. Consequently, measurement information may collapse as it passes through the network, and intermediate estimates can become detached from the raw measurements. This becomes especially problematic in practical lensless reconstruction, where the effective forward model varies across the field of view and is only approximately calibrated. In this setting, whether inversion is applied after this collapse or refinement follows an inversion without measurement-domain feedback, mismatch-induced errors can manifest as spatially structured artifacts and persist through later stages.
A key opportunity in this setting is to leverage physics priors and learned priors jointly throughout reconstruction, rather than confining physical consistency checks to a single stage. In optical systems with broad PSFs and long-range mixing, preserving measurement-domain cues while refining image-domain representations provides complementary information for recovering fine details. To this end, we introduce
IFIN
, a bidirectional reconstruction framework that interleaves differentiable forward projections with learnable inverse updates within an encoder–decoder hierarchy, enabling stable, physics-consistent reconstruction under large-footprint PSFs and shift-variant degradations. IFIN further learns a shift-variant PSF field end-to-end, improving robustness to calibration mismatch and enabling blind recovery when PSFs are inaccurate or unavailable.
Compared to the prior state of the art, IFIN improves PSNR on the three lensless benchmarks by +1.63 dB (DiffuserCam
[
34
]
), +0.65 dB (
WiderCam
, our newly introduced lensless benchmark), and +2.58 dB (MultiWienerNet
[
49
]
). Beyond lensless imaging, IFIN remains competitive on simulated Gaussian deblurring and achieves strong gains on inline holography reconstruction, demonstrating that the proposed bidirectional forward–inverse integration generalizes beyond a single modality and can be applied to a broader class of inverse problems.
Our main contributions are as follows:
1.
We propose IFIN, a reconstruction framework that embeds
bidirectional
forward–inverse guidance at
every
encoder–decoder scale, repeatedly exchanging measurement- and image-domain cues for physics- and data-driven refinement under long-range mixing.
2.
Within this framework, IFIN jointly learns a shift-variant PSF field, shared by both operators across scales, for robustness to calibration mismatch and blind recovery.
3.
IFIN achieves state-of-the-art performance on three lensless benchmarks and further validates the proposed approach on simulated Gaussian deblurring and inline holography.

## Method

The proposed IFIN adopts an encoder–decoder backbone equipped with
Integrated Forward–Inverse Blocks (IFIBs) at every scale
(
Fig.
1
). The encoder progressively downsamples the input measurement and coarse estimation to capture coarse-scale coupling and long-range interactions, while the decoder upsamples features back to the native resolution to recover fine details. At each resolution, an IFIB couples
a Forward System Operator (FSO), which maps image-domain features to the
measurement domain, with an Inverse System Operator (ISO), which restores
image-domain features from the measurement. Across scales, both operators condition on and jointly refine a learnable PSF field, enforcing forward–inverse consistency while propagating physically meaningful residual signals throughout the network.
Figure 1
:
Overall architecture of IFIN. The network follows an encoder–decoder structure, where Integrated Forward–Inverse Blocks (IFIBs) are inserted at each scale to jointly apply the Forward System Operator (FSO) and Inverse System Operator (ISO). A shared learnable PSF field guides both operators, ensuring forward–inverse consistency across scales. Notation:
n
n
indexes the per-scale PSF embedding
h
n
h_{n}
(one per scale, shared across the hierarchy);
(
n
)
(n)
denotes the stage-
n
n
representations (
x
(
n
)
,
y
(
n
)
x^{(n)},y^{(n)}
) updated across the hierarchy.
At the input stage, a coarse estimation is obtained by applying
the ISO to the measurement. The pair
(measurement, coarse estimation)
is then propagated as two coupled streams through the encoder–decoder
hierarchy. Within each IFIB, the FSO and ISO exchange features
bidirectionally, jointly enforcing consistency across the measurement
and image domains.
4.1
Learnable PSF Field
IFIN incorporates a learnable PSF representation that provides explicit system awareness to both the FSO and the ISO. The PSF field is parameterized as
k
=
s
2
k{=}s^{2}
kernels covering local regions of the image. When
s
=
1
s{=}1
, the PSF field reduces to a single global kernel. Kernels can be initialized from calibrated measurements, a single reference PSF, or random patterns, and are jointly optimized end-to-end within the network. In our experiments we use
k
∈
{
1
,
4
,
9
,
16
}
k\in\{1,4,9,16\}
for DiffuserCam and
k
=
9
k{=}9
for WiderCam and MultiWienerNet (MWNet), with cropped PSF supports of
270
×
270
270\times 270
,
135
×
135
135\times 135
, and
320
×
224
320\times 224
for the three lensless benchmarks (and
101
×
101
101\times 101
for Gaussian deblurring).
A compact PSF encoder maps the field to multi-scale embeddings
{
h
n
}
\{h_{n}\}
, where each
h
n
h_{n}
is shared between the encoder and decoder IFIBs at scale
n
n
and injected into both the FSO and the ISO, maintaining physical consistency across the hierarchy. Because the same PSF field must explain both how features generate measurements (FSO) and how measurements invert to sharp images (ISO), the PSFs are constrained by complementary supervision in both domains, which improves identifiability, discourages degenerate kernels, and aids blind PSF estimation.
We normalize the PSF to obtain unit DC gain in both the FSO and the ISO. This stabilizes the physics operators and prevents scale drift. We further impose a weak non-negativity regularizer on the PSF by penalizing negative values during training, guiding the learned kernels toward physically plausible solutions.
4.2
Integrated Forward–Inverse Block (IFIB)
The IFIB is the fundamental unit of IFIN, coupling forward and inverse imaging at each scale through two parallel operators, FSO and ISO (
Fig.
2
), both tied to the system’s physics. They can be configured as shift-invariant when degradations are approximately uniform, or spatially varying for more complex degradations, letting IFIN balance efficiency and fidelity.
Figure 2
:
(a) Schematic of the forward–inverse pairing in the IFIB. (b,c)
Single-PSF setting:
FSO uses 2D convolution; ISO uses Wiener-like deconvolution. (d,e)
PSF-field setting:
FSO uses a single representative (averaged) PSF from the PSF field; ISO applies region-wise deconvolution blended by learnable region-of-interest (ROI) maps.
Forward System Operator (FSO).
The FSO simulates how the current image-domain estimate
x
x
would be formed by
the imaging system, producing a forward projection in the measurement domain.
By default, we use 2D linear convolution with zero padding via a single
PSF
h
h
:
y
~
​
[
i
,
j
]
=
(
x
∗
h
)
​
[
i
,
j
]
.
\tilde{y}[i,j]\;=\;(x*h)[i,j].
(3)
The projection
y
~
\tilde{y}
is a measurement-domain proxy induced by the current image representation. Inside each IFIB it augments the measurement-domain features to improve their consistency with the forward model and to better support the subsequent inverse update. For efficiency under large-footprint PSFs, we compute it in the frequency domain with zero padding and crop back to the native resolution.
When the PSF field provides multiple kernels, the FSO uses a single representative (averaged) kernel for this projection. Because its role is to supply a stable measurement-consistency cue rather than to synthesize a high-fidelity measurement, an averaged kernel reduces model mismatch and avoids the cost of region-wise forward projection. In contrast, the ISO carries out the locally sensitive deconvolution and therefore applies region-wise inversion with ROI blending (below) to capture field-dependent degradation.
Inverse System Operator (ISO).
The ISO restores a sharp estimate from the degraded measurement via Wiener-like
deconvolution with a learnable frequency-dependent regularizer.
For PSF
h
h
, letting
Y
⁡
(
u
,
v
)
=
ℱ
​
{
W
⋅
P
r
​
p
​
y
}
​
(
u
,
v
)
Y(u,v)=\mathcal{F}\!\{\,W\cdot P_{rp}y\,\}(u,v)
and
H
⁡
(
u
,
v
)
=
ℱ
​
{
h
}
​
(
u
,
v
)
H(u,v)=\mathcal{F}\{h\}(u,v)
, where
ℱ
​
{
⋅
}
\mathcal{F}\{\cdot\}
denotes the Fourier transform,
P
r
​
p
P_{rp}
is replicate padding, and
W
W
is a mild Gaussian window used to mitigate
wrap-around artifacts during deconvolution
[
22
]
, we compute:
X
^
​
(
u
,
v
)
=
H
∗
​
(
u
,
v
)
|
H
⁡
(
u
,
v
)
|
2
+
ϵ
⁡
(
u
,
v
)
​
Y
​
(
u
,
v
)
,
ϵ
⁡
(
u
,
v
)
≥
0
,
\widehat{X}(u,v)\;=\;\frac{H^{*}(u,v)}{|H(u,v)|^{2}+\epsilon(u,v)}\,Y(u,v),\qquad\epsilon(u,v)\geq 0,
(4)
and set
x
^
=
ℱ
−
1
​
{
X
^
}
\hat{x}=\mathcal{F}^{-1}\{\widehat{X}\}
.
Here,
ϵ
⁡
(
u
,
v
)
\epsilon(u,v)
is a learnable 2D parameterization refined during training,
with non-negativity enforced by a ReLU. After the inverse Fourier transform, we crop the result back to the native resolution, matching the forward projection size.
Within each IFIB, this reconstructed estimate is forwarded as an augmentation
signal to the subsequent image-stream refinement.
When the PSF field provides multiple kernels, we can apply the same inverse
Eq.
4
region-wise using PSFs
{
h
r
}
r
=
1
k
\{h_{r}\}_{r=1}^{k}
and
regularizers
{
ϵ
r
}
r
=
1
k
\{\epsilon_{r}\}_{r=1}^{k}
, and blend the reconstructions
with normalized ROI weights to capture local variability:
x
^
​
[
i
,
j
]
=
∑
r
=
1
k
w
r
​
[
i
,
j
]
​
ℱ
−
1
​
{
X
^
r
}
​
[
i
,
j
]
,
\hat{x}[i,j]\;=\;\sum_{r=1}^{k}w_{r}[i,j]\,\mathcal{F}^{-1}\!\{\widehat{X}_{r}\}[i,j],
(5)
where
X
^
r
\widehat{X}_{r}
denotes the result of
Eq.
4
computed with the region-wise pair
(
h
r
,
ϵ
r
)
(h_{r},\epsilon_{r})
, and
{
w
r
}
r
=
1
k
\{w_{r}\}_{r=1}^{k}
are
k
k
learnable ROI maps. We initialize the ROI maps from
Gaussian kernels
{
g
r
}
r
=
1
k
\{g_{r}\}_{r=1}^{k}
centered at
p
r
p_{r}
, where
k
=
s
2
k{=}s^{2}
and
p
r
p_{r}
are
the centers of an
s
×
s
s\times s
grid partitioning the input measurement:
g
r
​
[
i
,
j
]
=
exp
⁡
(
−
‖
(
i
,
j
)
−
p
r
‖
2
2
2
​
σ
r
2
)
,
w
r
​
[
i
,
j
]
=
g
r
​
[
i
,
j
]
∑
q
=
1
k
g
q
​
[
i
,
j
]
,
g_{r}[i,j]\;=\;\exp\!\Big(-\tfrac{\|(i,j)-p_{r}\|_{2}^{2}}{2\sigma_{r}^{2}}\Big),\qquad w_{r}[i,j]\;=\;\frac{g_{r}[i,j]}{\sum_{q=1}^{k}g_{q}[i,j]},
(6)
∑
r
=
1
k
w
r
​
[
i
,
j
]
=
1
∀
(
i
,
j
)
,
\sum_{r=1}^{k}w_{r}[i,j]\;=\;1\ \ \forall(i,j),
(7)
where
σ
r
\sigma_{r}
is the width of the
r
r
-th initialization Gaussian.
Integrated Forward–Inverse.
The hallmark of the IFIB is the bidirectional exchange between FSO and ISO.
At each stage
(
n
)
{(n)}
, the FSO produces a measurement-domain projection
y
~
(
n
)
=
FSO
⁡
(
x
(
n
)
,
h
n
)
\tilde{y}^{(n)}=\mathrm{FSO}(x^{(n)};h_{n})
from the current image-domain
representation, while the ISO produces an image-domain estimate (a feature map)
x
~
(
n
)
=
ISO
⁡
(
y
(
n
)
,
h
n
)
\tilde{x}^{(n)}=\mathrm{ISO}(y^{(n)};h_{n})
from the current
measurement-domain representation. These cross-domain outputs are then used as
augmentation signals that are fused into the next updates of both streams:
y
(
n
+
1
)
=
ϕ
θ
y
​
(
α
F
(
n
)
⋅
y
(
n
)
+
β
F
(
n
)
⋅
y
~
(
n
)
)
,
y^{(n+1)}=\phi_{\theta}^{y}\!\left(\alpha_{F}^{(n)}\cdot y^{(n)}+\beta_{F}^{(n)}\cdot\tilde{y}^{(n)}\right),
(8)
x
(
n
+
1
)
=
ϕ
θ
x
​
(
α
I
(
n
)
⋅
x
(
n
)
+
β
I
(
n
)
⋅
x
~
(
n
)
)
,
x^{(n+1)}=\phi_{\theta}^{x}\!\left(\alpha_{I}^{(n)}\cdot x^{(n)}+\beta_{I}^{(n)}\cdot\tilde{x}^{(n)}\right),
(9)
where
ϕ
θ
y
\phi_{\theta}^{y}
and
ϕ
θ
x
\phi_{\theta}^{x}
each consist of a sequence of convolutional layers, and
α
F
(
n
)
,
β
F
(
n
)
\alpha_{F}^{(n)},\beta_{F}^{(n)}
and
α
I
(
n
)
,
β
I
(
n
)
\alpha_{I}^{(n)},\beta_{I}^{(n)}
are learnable per-channel scalar gates for the measurement and image streams at scale
n
n
.
Thus
y
~
(
n
)
\tilde{y}^{(n)}
updates the measurement representation before the next ISO step, while
x
~
(
n
)
\tilde{x}^{(n)}
drives the image-domain update and conditions the next FSO projection, forming a forward–inverse coupling across the hierarchy. Layer configurations and comprehensive FSO/ISO ablations are in the supplementary material.
