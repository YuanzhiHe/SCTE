# DRIFT: Difficulty-aware Rectified Flows for Through-plane MRI Super-Resolution

paper_id: arxiv:2607.16649v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

High-resolution (HR) isotropic Magnetic Resonance Imaging (MRI) provides crucial anatomical details for precise clinical diagnosis and quantitative neuroanatomical analysis. However, fundamental physical constraints such as the signal-to-noise ratio (SNR) and acquisition time often necessitate anisotropic acquisition, resulting in thick-slice volumes with low through-plane resolution. To bridge this gap, through-plane super-resolution (SR) approaches have emerged as a vital task to reconstruct isotropic volumes from anisotropic inputs.
Various approaches address through-plane SR under anisotropic acquisition, including self-supervision from in-plane and through-plane asymmetry
[
38
]
, alias-aware regression for thick-slice artifacts
[
25
]
, and synthesis-driven training for heterogeneous clinical protocols
[
10
]
. While these pipelines mitigate stair-step artifacts and improve reconstruction quality, regression-style estimators often favor conservative predictions and tend to be optimized for discrete target scales or protocol settings, which can limit generalization under the continuous slice-thickness variation encountered in practice. More broadly, continuously varying slice-thickness motivates SR formulations that represent images as continuous functions rather than relying on discrete grid-to-grid regression.
Implicit Neural Representations (INR) methods
[
6
,
12
]
represent images as continuous functions of spatial coordinates, with medical adaptations such as ArSSR
[
34
]
and SA-INR
[
32
]
extending this paradigm to 3D brain MRI. Despite their flexibility, these models face two challenges that matter for through-plane SR in MRI. First, they suffer from spectral bias
[
23
]
, where Multi-Layer Perceptron (MLP)-based continuous functions prioritize low-frequency structural components, which tends to under-recover high-frequency textures essential for clinical fidelity. Second, most existing arbitrary-scale models are trained using isotropic downsampling, which may not reflect the unique physics of MRI slice acquisition, where degradation is dominated by slice-profile governed through-plane integration
[
22
,
21
]
.
Generative approaches, particularly diffusion-based frameworks, have recently demonstrated superior performance in capturing complex textures. For instance, TPDM
[
14
]
utilizes pre-trained perpendicular 2D diffusion models to improve 3D imaging quality. However, the prohibitive inference cost remains a significant bottleneck for real-time clinical use. While recent advancements such as ResShift
[
36
]
have reduced sampling steps by initiating the sampling process from the low-resolution (LR) image instead of noise, they utilize a fixed-step scheduler regardless of the degradation severity of the input. In contrast, AdaDiffSR
[
8
]
introduces region-aware dynamic acceleration, but its image-dependent perception modules introduce non-negligible computational overhead and remain fundamentally detached from the underlying physical degradation (
i.e
.
, slice-thickness) inherent in MRI acquisition.
We propose
D
ifficulty-aware
R
ectif
I
ed
F
low for
T
hrough-plane SR (DRIFT), a two-stage framework for reconstructing a specified target resolution from inputs with continuous slice-thicknesses (Fig.
1
). DRIFT conditions both stages on the input and target slice thicknesses, allowing a single model to adapt its reconstruction process across different through-plane degradation levels. In Stage 1, DRIFT uses an Anatomical Projection Network (APN) to map LR patches to a coarse HR manifold, providing a structured initialization that shortens the subsequent refinement trajectory. In Stage 2, DRIFT refines high-frequency details using rectified flow, enabling deterministic inference with a reduced number of function evaluations (NFEs)
[
18
,
17
]
.
During training, DRIFT simulates thickness-diverse anisotropic inputs using a slice-profile model grounded in the Shinnar-Le Roux (SLR) algorithm
[
21
]
which mirrors the slice-selection process in MRI acquisition. We sample slice-thicknesses from common clinical ranges
[
11
,
10
]
and degradation axis, and apply an SLR-derived slice profile along the chosen axis to model through-plane signal integration. We then resample to the thick-slice grid and map the result back to the HR grid with nearest-neighbor interpolation to reproduce stair-step artifact morphology. To encourage thickness-consistent outputs across conditions, we introduce the Consistent Endpoint Trajectory Alignment (CETA) loss using proximal thickness pairs (Fig.
2
).
At test time, DRIFT uses a Physics-Aware Difficulty (PAD) metric computed from slice-thickness metadata to drive an Adaptive Integration Scheduler (AIS), which allocates ordinary differential equation (ODE) steps without image-dependent decision modules
[
8
]
(Fig.
1
). Our contributions are summarized as follows:
•
We formulate input slice-thickness-continuous through-plane MRI SR as a two-stage anatomical projection-to-transport problem, where a thickness-conditioned APN provides deterministic and spatially correlated slice-wise initialization that shortens the residual rectified-flow trajectory.
•
We introduce an MRI protocol-aware conditioning and inference mechanism: both stages are conditioned on input and target inverse thickness, while PAD/AIS maps slice-thickness-induced bandwidth deficit to adaptive ODE step budgets without auxiliary image-difficulty networks.
•
We propose CETA, a proximal-thickness endpoint alignment loss that regularizes rectified-flow trajectories across neighboring slice-thickness conditions and improves over naive random-pair consistency.
•
We validate DRIFT on public and real thick-slice MRI datasets with comprehensive reconstruction, ablation, standardized efficiency, zero-shot clinical transfer, and through-plane continuity analyses.
Figure 1
:
Conceptual overview of DRIFT inference.
(a) DRIFT initializes rectified flow from a coarse manifold estimate
z
z
rather than noise, shortening the residual transport trajectory.
(b) AIS allocates ODE steps using the PAD metric, assigning more steps to thick-slice hard cases and fewer steps to thin-slice easy cases.
Figure 2
:
Training pipeline of DRIFT. Stage 1 uses an Anatomical Projection Network (APN) to project LR patches to a coarse HR manifold conditioned on inverse thickness
τ
\tau
. Stage 2 refines textures via rectified flow by predicting a straight-line velocity field. We enforce Consistent Endpoint Trajectory Alignment (CETA) by regularizing trajectories between proximal thickness pairs (
T
j
=
T
i
−
Δ
​
T
T_{j}=T_{i}-\Delta T
,
Δ
​
T
=
1
\Delta T=1
mm). Notation is summarized in the figure legend.

## Method

We propose DRIFT (Difficulty-aware RectifIed Flow for Through-plane SR), a two-stage, slice-thickness-conditioned framework for through-plane MRI SR.
For the
p
p
-th training patch and a slice-thickness condition indexed by
i
i
with input thickness
T
i
T_{i}
, DRIFT reconstructs an isotropic target at thickness
T
hr
T_{\mathrm{hr}}
via
𝐱
p
,
i
→
Stage 1
APN
𝐳
p
,
i
→
Stage 2
Rectified Flow
𝐲
~
p
,
i
.
\mathbf{x}_{p,i}\xrightarrow[\text{Stage 1}]{\text{APN}}\mathbf{z}_{p,i}\xrightarrow[\text{Stage 2}]{\text{Rectified Flow}}\tilde{\mathbf{y}}_{p,i}.
(1)
Here,
𝐱
p
,
i
∈
ℝ
H
×
W
\mathbf{x}_{p,i}\in\mathbb{R}^{H\times W}
denotes the LR input patch under condition
i
i
,
𝐲
p
∈
ℝ
H
×
W
\mathbf{y}_{p}\in\mathbb{R}^{H\times W}
denotes the corresponding HR target patch at
T
hr
T_{\mathrm{hr}}
,
𝐳
p
,
i
\mathbf{z}_{p,i}
is a coarse anatomical estimate, and
𝐲
~
p
,
i
\tilde{\mathbf{y}}_{p,i}
is the final output.
DRIFT is implemented as a 2D slice wise model. Each volume is reconstructed by applying the same thickness-conditioned operator to slices orthogonal to the degraded axis and then composing the predicted slices back into a 3D volume. Therefore, DRIFT does not rely on explicit 3D convolutions or cross-slice communication. Its volumetric continuity instead comes from deterministic APN initialization and shared deterministic rectified flow refinement across neighboring band limited slices.
3.1
Slice-profile based thick-slice simulation
Through-plane degradation in thick-slice MRI is largely governed by slice-profile induced signal integration along the slice axis.
To approximate clinical slice selection, we synthesize LR inputs of thickness
T
i
T_{i}
from HR scans at thickness
T
hr
T_{\mathrm{hr}}
using an SLR-derived slice-profile model
[
21
]
.
Volume-level simulation.
Let
𝐘
∈
ℝ
H
×
W
×
D
\mathbf{Y}\in\mathbb{R}^{H\times W\times D}
denote an HR 3D volume sampled on an isotropic grid with spacing
T
hr
T_{\mathrm{hr}}
.
At each iteration, we sample a degradation axis
a
∈
{
x
,
y
,
z
}
a\in\{x,y,z\}
and apply a thickness-specific 1D slice-profile kernel
h
T
i
h_{T_{i}}
by convolving
𝐘
\mathbf{Y}
only along axis
a
a
(leaving the other axes unchanged), yielding a slice-profile integrated volume
𝐘
′
\mathbf{Y}^{\prime}
.
We then resample only along axis
a
a
from spacing
T
hr
T_{\mathrm{hr}}
to
T
i
T_{i}
(allowing non-integer factors), and represent the resulting thick-slice volume on the HR grid to form paired training data with
𝐘
\mathbf{Y}
.
This procedure reproduces the characteristic stair-step morphology observed in reformatted thick-slice MRI.
Implementation details are provided in the supplementary material (Sec. S1).
Patch extraction.
From the simulated thick-slice volume (represented on the HR grid) and the HR volume
𝐘
\mathbf{Y}
, we extract 2D slices orthogonal to the degraded axis
a
a
and crop aligned patch pairs, corresponding to
𝐱
p
,
i
\mathbf{x}_{p,i}
and
𝐲
p
\mathbf{y}_{p}
defined above.
3.2
Slice-Thickness Conditioning
We condition DRIFT on slice-thickness using the inverse thickness
τ
=
1
/
T
\tau=1/T
, which proxies the effective through-plane bandwidth under slice selection
[
21
,
2
]
.
For condition
i
i
, we set
τ
i
=
1
/
T
i
\tau_{i}=1/T_{i}
for the input thickness and
τ
hr
=
1
/
T
hr
\tau_{\mathrm{hr}}=1/T_{\mathrm{hr}}
for the target thickness. Using
τ
\tau
reduces the scale disparity between input and target thickness values over clinically common ranges and improves numerical conditioning.
We embed
τ
i
\tau_{i}
and
τ
hr
\tau_{\mathrm{hr}}
using two lightweight, unshared MLP encoders, concatenate their embeddings, and map the result to a thickness-conditioning vector
𝐜
i
∈
ℝ
d
c
\mathbf{c}_{i}\in\mathbb{R}^{d_{c}}
(we use
d
c
=
512
d_{c}=512
):
𝐜
i
=
MLP
cond
(
[
MLP
in
(
τ
i
)
∥
MLP
tgt
(
τ
hr
)
]
)
,
\mathbf{c}_{i}=\mathrm{MLP}_{\mathrm{cond}}\!\left(\left[\mathrm{MLP}_{\mathrm{in}}(\tau_{i})\;\|\;\mathrm{MLP}_{\mathrm{tgt}}(\tau_{\mathrm{hr}})\right]\right),
(2)
where
∥
\|\,
denotes concatenation,
MLP
in
\mathrm{MLP}_{\mathrm{in}}
and
MLP
tgt
\mathrm{MLP}_{\mathrm{tgt}}
encode the input and target inverse thickness, respectively, and
MLP
cond
\mathrm{MLP}_{\mathrm{cond}}
fuses the two embeddings into the final conditioning vector
𝐜
i
\mathbf{c}_{i}
.
In the main experiments,
T
hr
T_{\mathrm{hr}}
is fixed to the native isotropic resolution of each dataset, while the architecture itself accepts both input and target thickness embeddings. Target-continuous operation therefore requires sampling target thicknesses during training, which we examine in the supplementary material.
We inject
𝐜
i
\mathbf{c}_{i}
into APN residual blocks via Adaptive Group Normalization (AdaGN)
[
7
]
:
AdaGN
⁡
(
𝐡
,
𝐜
)
=
𝜸
⁡
(
𝐜
)
⊙
GN
⁡
(
𝐡
)
+
𝜷
⁡
(
𝐜
)
,
\mathrm{AdaGN}(\mathbf{h},\mathbf{c})=\boldsymbol{\gamma}(\mathbf{c})\odot\mathrm{GN}(\mathbf{h})+\boldsymbol{\beta}(\mathbf{c}),
(3)
where
𝐡
∈
ℝ
C
×
H
×
W
\mathbf{h}\in\mathbb{R}^{C\times H\times W}
is a feature map,
GN
\mathrm{GN}
denotes Group Normalization, and
(
𝜸
⁡
(
𝐜
)
,
𝜷
⁡
(
𝐜
)
)
∈
ℝ
2
×
C
(\boldsymbol{\gamma}(\mathbf{c}),\boldsymbol{\beta}(\mathbf{c}))\in\mathbb{R}^{2\times C}
are channel-wise scale and shift predicted from
𝐜
\mathbf{c}
by a learned linear projection. Architectural details of the MLPs are provided in the supplementary material (Sec. S4.1).
3.3
Stage 1: Structural Manifold Projection
Stage 1 is designed to shorten the rectified-flow transport in Stage 2. Given an LR patch
𝐱
p
,
i
\mathbf{x}_{p,i}
acquired under thickness condition
i
i
(input thickness
T
i
T_{i}
) and the thickness-conditioning vector
𝐜
i
\mathbf{c}_{i}
, the Anatomical Projection Network (APN)
f
ϕ
f_{\phi}
produces a coarse HR estimate
𝐳
p
,
i
\mathbf{z}_{p,i}
at the target thickness
T
hr
T_{\mathrm{hr}}
:
𝐳
p
,
i
=
f
ϕ
​
(
𝐱
p
,
i
,
𝐜
i
)
.
\mathbf{z}_{p,i}=f_{\phi}(\mathbf{x}_{p,i},\mathbf{c}_{i}).
(4)
We refer to this step as structural manifold projection because APN learns a thickness-conditioned mapping that projects samples from the thick-slice input space onto a coarse HR space anchored at
T
hr
T_{\mathrm{hr}}
. This projection reduces the remaining distance to the target HR distribution, allowing Stage 2 to focus on refining residual high-frequency details with fewer ODE steps.
Beyond reducing the transport distance, APN also provides an image-conditioned initial state for slice-wise generative refinement. Because neighboring thick-slice inputs share anatomical content and are processed by the same deterministic APN, their coarse HR states remain spatially correlated before Stage 2. This differs from slice-wise stochastic generative reconstruction, where independent noise initialization can introduce slice-to-slice variations unless additional noise-correlation heuristics are used.
Through-plane SR is ill-posed, and multiple HR solutions can explain the same thick-slice observation. Under pixel-wise regression, predictors tend to average over plausible solutions, resulting in smooth yet structurally consistent outputs. Accordingly, Stage 1 does not aim to recover the full HR texture distribution; instead, it provides a stable anatomical estimate close to the target HR manifold, which Stage 2 subsequently refines to recover sharp textures and boundaries.
We train APN with a reconstruction objective that combines the Charbonnier penalty
[
4
]
and SSIM
[
33
]
:
ℒ
recon
=
ℒ
Char
​
(
𝐳
p
,
i
,
𝐲
p
)
+
λ
ssim
​
ℒ
SSIM
​
(
𝐳
p
,
i
,
𝐲
p
)
,
\mathcal{L}_{\mathrm{recon}}=\mathcal{L}_{\mathrm{Char}}(\mathbf{z}_{p,i},\mathbf{y}_{p})+\lambda_{\mathrm{ssim}}\mathcal{L}_{\mathrm{SSIM}}(\mathbf{z}_{p,i},\mathbf{y}_{p}),
(5)
where
λ
ssim
\lambda_{\mathrm{ssim}}
weights the SSIM term (we use
λ
ssim
=
0.5
\lambda_{\mathrm{ssim}}=0.5
).
3.4
Stage 2: Difficulty-Aware Rectified Flow
Stage 2 refines high-frequency details that are under-recovered by Stage 1 by learning a rectified-flow velocity field
[
18
,
17
]
. During Stage 2 training, we freeze the Stage 1 APN so that the velocity network learns to refine a stable coarse estimate rather than a moving target. The velocity network takes the intermediate image state
𝐬
p
,
i
​
(
t
)
\mathbf{s}_{p,i}(t)
as input, while time
t
t
and slice-thickness information modulate intermediate features via AdaGN (Sec.
3.2
).
For training, we sample a normalized time
t
∈
[
0
,
1
]
t\in[0,1]
and define a straight path between the Stage 1 output
𝐳
p
,
i
\mathbf{z}_{p,i}
and the HR target
𝐲
p
\mathbf{y}_{p}
:
𝐬
p
,
i
​
(
t
)
=
(
1
−
t
)
​
𝐳
p
,
i
+
t
​
𝐲
p
,
𝐮
p
,
i
=
𝐲
p
−
𝐳
p
,
i
,
\mathbf{s}_{p,i}(t)=(1-t)\mathbf{z}_{p,i}+t\mathbf{y}_{p},\qquad\mathbf{u}_{p,i}=\mathbf{y}_{p}-\mathbf{z}_{p,i},
(6)
where
𝐮
p
,
i
\mathbf{u}_{p,i}
is the constant target velocity along the path. Given
𝐬
p
,
i
​
(
t
)
\mathbf{s}_{p,i}(t)
, the velocity network
v
θ
v_{\theta}
predicts the rectified-flow velocity, with
t
t
and slice-thickness entering only through conditioning. Concretely, we compute a time embedding from a sinusoidal time embedding
[
31
]
, concatenate it with the thickness-conditioning vector
𝐜
i
\mathbf{c}_{i}
(Sec.
3.2
), and map the concatenation to a joint time–thickness conditioning vector:
𝐞
t
=
MLP
time
(
SinEmb
(
t
)
)
,
𝐜
t
,
i
=
MLP
tt
(
[
𝐞
t
∥
𝐜
i
]
)
,
\mathbf{e}_{t}=\mathrm{MLP}_{\mathrm{time}}\!\left(\mathrm{SinEmb}(t)\right),\qquad\mathbf{c}_{t,i}=\mathrm{MLP}_{\mathrm{tt}}\!\left([\mathbf{e}_{t}\;\|\;\mathbf{c}_{i}]\right),
(7)
where
∥
\|\,
denotes concatenation and
MLP
time
\mathrm{MLP}_{\mathrm{time}}
and
MLP
tt
\mathrm{MLP}_{\mathrm{tt}}
are lightweight MLPs (architectural details are provided in the supplementary material Sec. S4.1). The predicted velocity is
𝐯
p
,
i
​
(
t
)
=
v
θ
​
(
𝐬
p
,
i
​
(
t
)
,
𝐜
t
,
i
)
,
\mathbf{v}_{p,i}(t)=v_{\theta}\!\left(\mathbf{s}_{p,i}(t);\mathbf{c}_{t,i}\right),
(8)
where
𝐬
p
,
i
​
(
t
)
\mathbf{s}_{p,i}(t)
is the only network input, and
𝐜
t
,
i
\mathbf{c}_{t,i}
is injected into residual blocks via AdaGN.
We train
v
θ
v_{\theta}
to match
𝐮
p
,
i
\mathbf{u}_{p,i}
using a Huber objective with threshold
δ
=
0.1
\delta=0.1
and an endpoint-biased timestep distribution
p
α
​
(
t
)
p_{\alpha}(t)
, inspired by the difficulty-aware timestep reweighting principle in RF++
[
13
]
:
ℒ
RF
=
𝔼
t
∼
p
α
​
(
t
)
​
[
Huber
δ
​
(
𝐯
p
,
i
​
(
t
)
,
𝐮
p
,
i
)
]
.
\mathcal{L}_{\mathrm{RF}}=\mathbb{E}_{t\sim p_{\alpha}(t)}\left[\mathrm{Huber}_{\delta}\!\left(\mathbf{v}_{p,i}(t),\mathbf{u}_{p,i}\right)\right].
(9)
To sample
t
t
, we draw
r
∼
𝒰
⁡
(
0
,
1
)
r\sim\mathcal{U}(0,1)
and apply
t
=
1
2
−
1
2
​
sgn
​
(
r
−
1
2
)
​
|
2
​
r
−
1
|
1
/
α
,
t=\frac{1}{2}-\frac{1}{2}\,\mathrm{sgn}\!\left(r-\tfrac{1}{2}\right)\,|2r-1|^{1/\alpha},
(10)
where
sgn
⁡
(
⋅
)
\mathrm{sgn}(\cdot)
is the sign function and
α
\alpha
controls the degree of endpoint emphasis (we use
α
=
2.0
\alpha=2.0
; see Sec. S8 in the supplementary material).
Here,
p
α
​
(
t
)
p_{\alpha}(t)
denotes the induced distribution of
t
t
under Eq.
10
.
This sampling allocates more probability mass to
t
≈
0
t\approx 0
and
t
≈
1
t\approx 1
, where refinement is most sensitive in our residual regime. Finally, we zero-initialize the output layer of
v
θ
v_{\theta}
so that
𝐯
p
,
i
​
(
t
)
≈
𝟎
\mathbf{v}_{p,i}(t)\approx\mathbf{0}
at the beginning of training, making Stage 2 initially behave as an identity refinement that preserves
𝐳
p
,
i
\mathbf{z}_{p,i}
and then progressively learns residual high-frequency details.
3.5
CETA: Consistent Endpoint Trajectory Alignment Loss
CETA regularizes thickness-consistent reconstructions for the same underlying anatomy. For a proximal thickness pair
(
i
,
j
)
(i,j)
generated from the same HR patch
𝐲
p
\mathbf{y}_{p}
, we set
T
j
=
T
i
−
Δ
​
T
T_{j}=T_{i}-\Delta T
with
Δ
​
T
=
1
\Delta T=1
mm. Here,
𝐳
p
,
k
\mathbf{z}_{p,k}
denotes the Stage 1 APN output for condition
k
k
(Eq. (
4
)), for
k
∈
{
i
,
j
}
k\in\{i,j\}
. We form endpoint proxies
𝐲
~
p
,
k
​
(
t
)
=
𝐳
p
,
k
+
𝐯
p
,
k
​
(
t
)
,
k
∈
{
i
,
j
}
,
\tilde{\mathbf{y}}_{p,k}(t)=\mathbf{z}_{p,k}+\mathbf{v}_{p,k}(t),\qquad k\in\{i,j\},
(11)
where
𝐯
p
,
k
​
(
t
)
\mathbf{v}_{p,k}(t)
is the predicted rectified-flow velocity (Eq. (
8
)). CETA minimizes the squared L2 distance between proxies:
ℒ
CETA
=
‖
𝐲
~
p
,
i
​
(
t
)
−
𝐲
~
p
,
j
​
(
t
)
‖
2
2
.
\mathcal{L}_{\mathrm{CETA}}=\left\|\tilde{\mathbf{y}}_{p,i}(t)-\tilde{\mathbf{y}}_{p,j}(t)\right\|_{2}^{2}.
(12)
Using a fixed proximal gap yields a chain of local constraints across the sampled thickness range, which propagates consistency beyond a single pair (
e.g
.
,
T
6
↔
T
5
↔
⋯
↔
T
1
T_{6}\leftrightarrow T_{5}\leftrightarrow\cdots\leftrightarrow T_{1}
). A very small
Δ
​
T
\Delta T
yields nearly identical targets and provides weak alignment signal, while a very large
Δ
​
T
\Delta T
forces alignment between substantially different trajectories and can destabilize training. We therefore use
Δ
​
T
=
1
\Delta T=1
mm as a practical balance. The Stage 2 objective is
ℒ
Stage2
=
ℒ
RF
+
λ
ceta
​
ℒ
CETA
\mathcal{L}_{\mathrm{Stage2}}=\mathcal{L}_{\mathrm{RF}}+\lambda_{\mathrm{ceta}}\mathcal{L}_{\mathrm{CETA}}
(we use
λ
ceta
=
1.0
\lambda_{\mathrm{ceta}}=1.0
. See Sec. S6 in the supplementary material).
3.6
PAD and AIS: Physics-Aware Adaptive Inference
Inference cost in Stage 2 is dominated by the number of velocity evaluations. Using a fixed step, therefore, wastes computation for thin-slice inputs and can under-refine thick-slice inputs. DRIFT addresses this by selecting the number of Euler steps
N
N
, equivalently the NFEs of the velocity network, directly from slice-thickness metadata.
Physics-Aware Difficulty (PAD).
Given the input thickness
T
i
T_{i}
and the fixed target thickness
T
hr
T_{\mathrm{hr}}
(Sec.
3
), thicker slices lose a larger fraction of through-plane frequency content due to slice-direction integration. We quantify this protocol difficulty by the normalized bandwidth deficit:
PAD
⁡
(
T
i
,
T
hr
)
=
1
−
T
hr
T
i
,
\mathrm{PAD}(T_{i},T_{\mathrm{hr}})=1-\frac{T_{\mathrm{hr}}}{T_{i}},
(13)
where
PAD
∈
[
0
,
1
)
\mathrm{PAD}\in[0,1)
, equals
0
0
when
T
i
=
T
hr
T_{i}=T_{\mathrm{hr}}
. PAD serves as a physics-aware degradation severity cue rather than an image difficulty score. It reflects the normalized through-plane bandwidth deficit given by
1
−
T
hr
/
T
i
1-T_{\mathrm{hr}}/T_{i}
, which increases as the input slice becomes thicker relative to the target resolution.
Adaptive Integration Scheduler (AIS).
AIS maps
PAD
⁡
(
T
i
,
T
hr
)
\mathrm{PAD}(T_{i},T_{\mathrm{hr}})
to the step budget:
N
=
clamp
⁡
(
⌊
N
max
⋅
PAD
⁡
(
T
i
,
T
hr
)
⌉
,
N
min
,
N
max
)
,
N=\mathrm{clamp}\!\Big(\big\lfloor N_{\max}\cdot\mathrm{PAD}(T_{i},T_{\mathrm{hr}})\big\rceil,\;N_{\min},\;N_{\max}\Big),
(14)
where
⌊
⋅
⌉
\lfloor\cdot\rceil
rounds to the nearest integer and
clamp
⁡
(
x
,
a
,
b
)
=
min
⁡
(
max
⁡
(
x
,
a
)
,
b
)
\mathrm{clamp}(x,a,b)=\min(\max(x,a),b)
. We set
N
min
=
0
N_{\min}=0
and
N
max
=
15
N_{\max}=15
(See Sec.S7 in the supplementary material). This allocation uses fewer NFEs for easier (thin-slice) cases and more NFEs for harder (thick-slice) cases, with negligible overhead since it depends only on slice-thickness metadata.
