# InverseNet: Benchmarking Operator Mismatch and Calibration Across Compressive Imaging Modalities

paper_id: arxiv:2603.04538v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Compressive imaging acquires fewer measurements than the Nyquist limit by exploiting signal structure, recovering the full signal through computational reconstruction.
This paradigm underlies diverse modalities including hyperspectral imaging via coded apertures
[
1
,
2
]
, video compressive sensing via temporal coding
[
3
,
4
]
, and single-pixel cameras via structured illumination
[
5
,
6
]
.
In all cases, reconstruction quality depends critically on knowledge of the forward measurement operator—the mapping from scene to measurements.
Yet a dangerous chasm separates research from reality.
Reconstruction algorithms are benchmarked with idealized forward operators, but deployed systems suffer
operator mismatch
: EfficientSCI
[
12
]
collapses from 35.39 to 14.81 dB—a 20.58 dB drop—under realistic 8-parameter mismatch.
For CASSI, mask misalignment of 0.5 px combined with 1% dispersion drift degrades PSNR by over 13 dB (
table
1
); for CACTI, spatial, temporal, and radiometric errors compound across 8 parameters; for single-pixel cameras, gain drift causes systematic errors.
These mismatches are ubiquitous yet systematically ignored in benchmarks.
The benchmark gap.
Just as CASP
[
26
]
transformed protein folding by forcing blind prediction against nature, computational imaging needs benchmarks that evaluate against physical reality.
Existing benchmarks—KAIST
[
17
]
for CASSI, the CACTI benchmark
[
4
]
—assume perfect operator knowledge, providing no information about mismatch robustness.
Antun et al.
[
27
]
showed deep learning solvers are unstable to adversarial perturbations; InverseNet operationalizes this for
physically realistic
operator mismatch.
Contributions.
We address this gap with InverseNet, which makes four contributions:
1.
Unified four-scenario protocol.
We define four scenarios—ideal (I), mismatched (II), oracle-corrected (III), and blind calibration (IV)—applicable across modalities. The I
→
\to
II gap quantifies mismatch sensitivity; II
→
\to
III quantifies calibration potential; IV measures practical recovery via self-supervised calibration.
2.
Cross-modality benchmark.
We evaluate 12 methods (4 CASSI, 4 CACTI, 4 SPC) spanning classical, plug-and-play, and deep learning approaches across 27 simulated scenes, producing over 360 experiments.
3.
Real hardware validation.
We validate simulation findings on 5 real CASSI scenes and 4 real CACTI scenes from publicly available hardware captures, confirming that mismatch patterns transfer to physical data.
4.
Open dataset.
All reconstruction arrays, per-scene metrics, and analysis code will be publicly released.
1
1
1
Code available upon acceptance.
All test data come from existing public datasets.
Our key findings include: (a) operator mismatch degrades deep learning methods by 10–21 dB while classical methods lose only 3–11 dB; (b) operator-aware architectures are simultaneously the most sensitive to mismatch and the most recoverable through calibration; (c) mask-oblivious architectures show zero calibration benefit; (d) CACTI exhibits the most severe degradation (up to 20.58 dB) due to its 8-parameter mismatch space; (e) dispersion mismatch in CASSI limits oracle recoverability due to fixed-step architectural assumptions.

## Method

3.1
Unified Four-Scenario Protocol
We define four evaluation scenarios that apply uniformly across all compressive imaging modalities.
Let
𝚽
\mathbf{\Phi}
denote the true (physical) forward operator and
𝚽
^
\hat{\mathbf{\Phi}}
the assumed (nominal) operator used during reconstruction.
•
Scenario I (Ideal):
𝐲
=
𝚽
^
​
𝐱
+
𝐧
\mathbf{y}=\hat{\mathbf{\Phi}}\mathbf{x}+\mathbf{n}
, reconstruct with
𝚽
^
\hat{\mathbf{\Phi}}
. Best-case performance with perfect operator knowledge.
•
Scenario II (Baseline):
𝐲
=
𝚽
​
𝐱
+
𝐧
\mathbf{y}=\mathbf{\Phi}\mathbf{x}+\mathbf{n}
, reconstruct with
𝚽
^
\hat{\mathbf{\Phi}}
. Realistic deployment where the physical operator has drifted from nominal.
•
Scenario III (Oracle):
𝐲
=
𝚽
​
𝐱
+
𝐧
\mathbf{y}=\mathbf{\Phi}\mathbf{x}+\mathbf{n}
, reconstruct with
𝚽
\mathbf{\Phi}
. Upper bound achievable through perfect calibration.
•
Scenario IV (Blind Calibration):
𝐲
=
𝚽
​
𝐱
+
𝐧
\mathbf{y}=\mathbf{\Phi}\mathbf{x}+\mathbf{n}
, reconstruct with
𝚽
~
\tilde{\mathbf{\Phi}}
estimated via grid search over mismatch parameters using a self-supervised objective (measurement residual for geometric mismatch, reconstruction sparsity for radiometric mismatch). Practical calibration without ground truth.
This protocol yields two diagnostic metrics per method:
Δ
deg
\displaystyle\Delta_{\text{deg}}
=
PSNR
I
−
PSNR
II
\displaystyle=\text{PSNR}_{\text{I}}-\text{PSNR}_{\text{II}}
(mismatch degradation)
,
\displaystyle\text{(mismatch degradation)},
(1)
Δ
rec
\displaystyle\Delta_{\text{rec}}
=
PSNR
III
−
PSNR
II
\displaystyle=\text{PSNR}_{\text{III}}-\text{PSNR}_{\text{II}}
(oracle recovery)
,
\displaystyle\text{(oracle recovery)},
(2)
and the
recovery ratio
ρ
=
Δ
rec
/
Δ
deg
∈
[
0
,
1
]
\rho=\Delta_{\text{rec}}/\Delta_{\text{deg}}\in[0,1]
, which measures what fraction of the mismatch loss can be recovered through calibration.
3.2
CASSI: Coded Aperture Snapshot Spectral Imaging
Forward model.
CASSI acquires a 2D measurement
𝐲
∈
ℝ
H
×
W
′
\mathbf{y}\in\mathbb{R}^{H\times W^{\prime}}
of a 3D hyperspectral cube
𝐱
∈
ℝ
H
×
W
×
Λ
\mathbf{x}\in\mathbb{R}^{H\times W\times\Lambda}
through a coded aperture mask
𝐌
∈
{
0
,
1
}
H
×
W
\mathbf{M}\in\{0,1\}^{H\times W}
followed by a dispersive prism.
The measurement at pixel
(
i
,
j
)
(i,j)
is:
y
⁡
(
i
,
j
)
=
∑
λ
=
1
Λ
M
⁡
(
i
,
j
−
d
⁡
(
λ
)
)
⋅
x
⁡
(
i
,
j
,
λ
)
+
n
⁡
(
i
,
j
)
,
y(i,j)=\sum_{\lambda=1}^{\Lambda}M(i,j-d(\lambda))\cdot x(i,j,\lambda)+n(i,j),
(3)
where
d
⁡
(
λ
)
d(\lambda)
is the dispersion shift for spectral band
λ
\lambda
and
W
′
=
W
+
(
Λ
−
1
)
⋅
s
W^{\prime}=W+(\Lambda-1)\cdot s
with dispersion step
s
s
.
Mismatch model.
We model CASSI operator mismatch as a 5-parameter perturbation combining mask misalignment and dispersion drift:
𝚽
=
𝒟
⁡
(
a
1
,
α
)
∘
𝒯
⁡
(
d
​
x
,
d
​
y
,
θ
)
∘
𝚽
^
,
\mathbf{\Phi}=\mathcal{D}(a_{1},\alpha)\circ\mathcal{T}(dx,dy,\theta)\circ\hat{\mathbf{\Phi}},
(4)
where
d
​
x
,
d
​
y
dx,dy
are subpixel translational shifts,
θ
\theta
is a rotational misalignment of the coded aperture mask,
a
1
a_{1}
is the dispersion slope (nominal
s
=
2.0
s=2.0
px/band), and
α
\alpha
is the dispersion axis angular offset.
We use
d
​
x
=
0.5
dx=0.5
px,
d
​
y
=
0.3
dy=0.3
px,
θ
=
0.1
∘
\theta=0.1^{\circ}
for mask misalignment, and
a
1
=
2.02
a_{1}=2.02
px/band (1% drift from nominal) and
α
=
0.15
∘
\alpha=0.15^{\circ}
for dispersion mismatch, representing moderate assembly and optical tolerances.
Reconstruction methods.
We evaluate four methods:
GAP-TV
[
7
]
: classical accelerated proximal gradient with TV regularization (100 iterations,
λ
TV
=
0.1
\lambda_{\text{TV}}=0.1
);
PnP-HSICNN
[
33
]
: plug-and-play GAP with HSI-SDeCNN deep denoiser (124 iterations: TV iters 0–82, HSICNN iters 83–123);
HDNet
[
11
]
: dual-domain deep network with spectral discrimination learning (pretrained);
MST-L
[
10
]
: mask-guided spectral transformer, large variant (2 stages, blocks
[
4
,
7
,
5
]
[4,7,5]
, pretrained).
Dataset.
We use 10 scenes from the KAIST TSA simulated dataset
[
17
]
, each consisting of a
256
×
256
×
28
256\times 256\times 28
hyperspectral cube spanning 450–650 nm.
Measurements are formed with a binary random mask (
s
=
2
s=2
pixels/band), yielding
256
×
310
256\times 310
detector images.
Low noise (
α
=
10
5
\alpha=10^{5}
photon peak,
σ
=
0.01
\sigma=0.01
read noise) isolates the effect of operator mismatch.
3.3
CACTI: Coded Aperture Compressive Temporal Imaging
Forward model.
CACTI acquires a single 2D snapshot
𝐲
∈
ℝ
H
×
W
\mathbf{y}\in\mathbb{R}^{H\times W}
encoding
B
B
high-speed video frames
𝐱
∈
ℝ
H
×
W
×
B
\mathbf{x}\in\mathbb{R}^{H\times W\times B}
through a dynamic coded aperture:
y
⁡
(
i
,
j
)
=
∑
b
=
1
B
C
b
​
(
i
,
j
)
⋅
x
⁡
(
i
,
j
,
b
)
+
n
⁡
(
i
,
j
)
,
y(i,j)=\sum_{b=1}^{B}C_{b}(i,j)\cdot x(i,j,b)+n(i,j),
(5)
where
C
b
∈
{
0
,
1
}
H
×
W
C_{b}\in\{0,1\}^{H\times W}
is the binary mask pattern for temporal frame
b
b
.
Mismatch model.
CACTI mismatch involves 8 parameters capturing spatial, temporal, and radiometric errors:
spatial shifts (
d
​
x
=
0.5
dx=0.5
px,
d
​
y
=
0.3
dy=0.3
px), rotation (
θ
=
0.1
∘
\theta=0.1^{\circ}
), temporal clock offset (
Δ
​
t
=
0.05
\Delta t=0.05
), duty cycle deviation (
η
=
0.95
\eta=0.95
), detector gain (
g
=
1.02
g=1.02
), offset (
o
=
0.002
o=0.002
), and measurement noise (
σ
n
=
1.0
\sigma_{n}=1.0
).
Reconstruction methods.
We evaluate four methods:
GAP-TV
[
7
]
: classical iterative with TV regularization;
PnP-FFDNet
[
16
]
: plug-and-play with FFDNet denoiser;
ELP-Unfolding
[
13
]
: ensemble learning priors driven deep unfolding network (pretrained);
EfficientSCI
[
12
]
: efficient deep learning for snapshot compressive imaging (pretrained).
Dataset.
We use 6 standard benchmark videos (
kobe
,
traffic
,
runner
,
drop
,
crash
,
aerial
) at
256
×
256
256\times 256
resolution with
B
=
8
B=8
temporal frames per snapshot, following the standard video compressive sensing evaluation protocol
[
4
]
.
3.4
SPC: Single-Pixel Camera
Forward model.
The single-pixel camera acquires
m
m
scalar measurements of an image
𝐱
∈
ℝ
n
\mathbf{x}\in\mathbb{R}^{n}
through structured illumination patterns:
𝐲
=
𝐀𝐱
+
𝐧
,
\mathbf{y}=\mathbf{A}\mathbf{x}+\mathbf{n},
(6)
where
𝐀
∈
ℝ
m
×
n
\mathbf{A}\in\mathbb{R}^{m\times n}
is the measurement matrix (typically Gaussian or Hadamard patterns) with compression ratio
m
/
n
m/n
.
Mismatch model.
We model SPC mismatch as exponential gain drift affecting the measurement rows:
𝚽
=
diag
(
e
−
α
⋅
𝐢
)
⋅
𝚽
^
,
\mathbf{\Phi}=\text{diag}\!\left(e^{-\alpha\cdot\mathbf{i}}\right)\cdot\hat{\mathbf{\Phi}},
(7)
where
α
=
0.0015
\alpha=0.0015
controls the drift rate and
𝐢
=
[
0
,
1
,
…
,
m
−
1
]
⊤
\mathbf{i}=[0,1,\ldots,m{-}1]^{\!\top}
indexes the measurement rows, modelling progressive detector gain decay during sequential acquisition.
Additional measurement noise
σ
y
=
0.03
\sigma_{y}=0.03
is applied.
Reconstruction methods.
We evaluate four methods:
FISTA-TV
[
8
]
: fast iterative shrinkage-thresholding with TV regularization (500 iterations,
λ
=
0.005
\lambda=0.005
);
PnP-DRUNet
[
34
]
: plug-and-play FISTA with DRUNet denoiser and sigma annealing (200 iterations, row-normalized operator);
ISTA-Net
[
14
]
: learned iterative shrinkage-thresholding network (pretrained);
HATNet
[
15
]
: dual-scale transformer for single-pixel imaging (pretrained).
Dataset.
We use the 11 standard Set11 test images (
Monarch
,
Parrots
,
barbara
,
boats
,
cameraman
,
fingerprint
,
flinstones
,
foreman
,
house
,
lena256
,
peppers256
) at
256
×
256
256\times 256
resolution with 25% sampling ratio.
3.5
Evaluation Metrics
We report three standard image quality metrics:
•
PSNR
(peak signal-to-noise ratio, dB): pixel-level fidelity, computed per-channel and averaged.
•
SSIM
(structural similarity index): perceptual structural quality
[
21
]
.
•
SAM
(spectral angle mapper, degrees): spectral fidelity, reported for CASSI only.
