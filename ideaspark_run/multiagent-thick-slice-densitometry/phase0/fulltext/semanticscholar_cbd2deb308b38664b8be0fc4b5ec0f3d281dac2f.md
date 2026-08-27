# Conjugate Gradient Unrolled Network with PSF Conditioning for Non-Diagonal Data Fidelity in CASSI Reconstruction

paper_id: semanticscholar:cbd2deb308b38664b8be0fc4b5ec0f3d281dac2f
tier: T2
source_used: html_arxiv
warning: none

## Intro

Coded aperture snapshot spectral imaging (CASSI) has emerged as a promising paradigm for acquiring three-dimensional hyperspectral images (HSIs) from a single two-dimensional compressed measurement
[
1
,
2
]
. By leveraging a coded aperture mask and dispersive optics, CASSI systems encode the spatial-spectral information of a scene into a snapshot measurement, enabling high-throughput data acquisition without the need for temporal scanning. The reconstruction of the original HSI from this compressed measurement constitutes a highly ill-posed inverse problem, which has attracted significant research attention in recent years
[
6
,
10
,
14
,
15
,
16
]
.
Deep unfolding methods have demonstrated remarkable success in CASSI reconstruction by combining the interpretability of optimization-based algorithms with the representational power of deep neural networks
[
13
,
17
,
14
,
15
,
18
,
16
]
. These methods formulate the reconstruction as an optimization problem and unroll the iterative solver into a fixed number of network stages, where each stage corresponds to one iteration with learnable components. Notable examples include ADMM-Net
[
19
]
, DGSMP
[
13
]
, DAUHST
[
14
]
, RDLUF
[
15
]
, and the recent Dual Prior Unfolding (DPU)
[
16
]
, which achieves state-of-the-art performance by jointly exploiting image priors and degradation-associated priors within an augmented Lagrangian framework. However, these methods universally adopt a mask-only forward model
𝐠
=
𝚽
​
𝐟
+
𝜺
\mathbf{g}=\mathbf{\Phi}\mathbf{f}+\bm{\varepsilon}
, implicitly assuming that the optical point spread function (PSF) is ideal (i.e., a Dirac delta function).
In practice, every optical system introduces spatially-varying and wavelength-dependent PSF degradation due to residual aberrations. In CASSI systems, optical aberrations such as coma, astigmatism, and field curvature cause the PSF to vary across the field of view and with wavelength. The mismatch between the assumed mask-only forward model and the actual PSF-degraded imaging process introduces systematic reconstruction errors that cannot be compensated by increasing network capacity alone.
The importance of PSF modeling in CASSI has been recognized in prior work along two distinct lines. On the physical modeling side, Song et al.
[
4
]
proposed a high-accuracy image formation model that explicitly incorporates the system PSF, estimated from calibration images via regularized least-squares, into the measurement matrix for traditional optimization-based reconstruction (e.g., total variation and non-local low-rank methods). Their work demonstrated that faithful PSF modeling significantly improves reconstruction quality over the simplified binary model. On the data-driven side, Yue et al.
[
5
]
computed the Huygens PSF from optical design parameters and used it both to generate realistic aberrated training data and as a conditional input to a cGAN-based reconstruction network, achieving improved robustness to optical aberrations without requiring physical calibration.
Despite these advances, a fundamental gap remains:
no existing method integrates a PSF-inclusive forward operator into a deep unfolding reconstruction framework
. Song et al.’s approach
[
4
]
pairs PSF-aware physics with traditional solvers that lack the representational power of learned priors. Yue et al.’s approach
[
5
]
leverages PSF information within a data-driven network but does not embed the PSF into an iterative physics-consistent optimization loop. Meanwhile, state-of-the-art deep unfolding methods achieve strong reconstruction performance but remain entirely PSF-agnostic. Bridging PSF-aware optical modeling with deep unfolding-based reconstruction is the central goal of this work.
The integration of PSF into a deep unfolding framework is not merely an engineering extension—it introduces a fundamental algorithmic challenge. In conventional unfolding methods such as DPU
[
16
]
, the forward model only accounts for coded aperture modulation and dispersive shifting, resulting in a diagonal normal matrix
𝚽
T
​
𝚽
\mathbf{\Phi}^{T}\mathbf{\Phi}
that admits a closed-form solution for the data-fidelity subproblem. However, when PSF information is incorporated into the measurement matrix, this diagonal structure is broken and no closed-form solution exists, necessitating a new solver for the data-fidelity subproblem.
To address these challenges, we propose a PSF-aware deep unfolding network for CASSI reconstruction that integrates the PSF-inclusive forward operator
[
𝐀
⁡
(
𝐟
)
]
c
=
H
c
∗
(
Φ
c
⊙
f
c
)
[\mathbf{A}(\mathbf{f})]_{c}=H_{c}*(\Phi_{c}\odot f_{c})
, following the physical imaging model of Song et al.
[
4
]
, into the iterative optimization structure of the DPU framework
[
16
]
. Since PSF inclusion renders the normal matrix
𝐀
T
​
𝐀
\mathbf{A}^{T}\mathbf{A}
non-diagonal, we introduce a differentiable conjugate gradient (CG) solver with
K
=
2
K=2
warm-started iterations. PSF information is further injected at the optimization level through two complementary pathways: (i) a global PSF embedding conditions the ADMM penalty parameter, and (ii) a per-wavelength PSF embedding generates wavelength-adaptive step sizes and gradient biases in the post-CG refinement step. A Monte Carlo PSF training strategy based on Zernike polynomial tolerance analysis exposes the network to realistic manufacturing-induced PSF variations.
We evaluate our method on the CAVE
[
31
]
and KAIST
[
32
]
datasets under field-dependent PSF degradation simulated from a real DD-CASSI optical design using Zemax. Simulation experiments show a 1.70 dB PSNR improvement over the larger PSF-agnostic baseline (DPU-B+) under nominal PSF conditions and 1.83 dB under Monte Carlo PSF conditions. And larger improvements are observed over other state-of-the-art unfolding methods (e.g., +3.14 dB over RDLUF, +2.39 dB over MST).
The main contributions of this work are summarized as follows:
•
We propose, to the best of our knowledge, the first PSF-aware deep unfolding framework for CASSI, integrating field-dependent, wavelength-dependent PSF into the physical operators of each unfolding stage. Unlike prior work that pairs PSF modeling with traditional solvers
[
4
]
or uses PSF as side information for data-driven networks
[
5
]
, our method embeds the PSF directly into the iterative optimization structure.
•
We introduce a
K
K
-step CG-unrolled data-fidelity solver as a necessary algorithmic consequence of the PSF-inclusive forward model, replacing the closed-form solution that becomes inapplicable when
𝐀
T
​
𝐀
\mathbf{A}^{T}\mathbf{A}
is non-diagonal. The CG solver is fully differentiable and supports end-to-end training.
•
We propose a wavelength-adaptive gradient refinement module that uses a learned per-wavelength PSF embedding to generate spectral-channel-specific step sizes and gradient biases for the post-CG update, enabling the network to apply per-channel adaptive convergence rates based on PSF severity.
•
We propose a Monte Carlo PSF training strategy based on Zernike polynomial tolerance analysis, exposing the network to realistic manufacturing-induced PSF variations and improving robustness to off-design PSFs.

## Method

3.1
Conventional CASSI Forward Model
The dual-disperser coded aperture snapshot spectral imaging (DD-CASSI) system captures a three-dimensional hyperspectral image
𝐅
∈
ℝ
H
×
W
×
Λ
\mathbf{F}\in\mathbb{R}^{H\times W\times\Lambda}
into a single two-dimensional measurement
𝐆
∈
ℝ
H
×
W
\mathbf{G}\in\mathbb{R}^{H\times W}
. In DD-CASSI, the first dispersive prism introduces wavelength-dependent spatial shifts, the coded aperture modulates the dispersed light, and the second prism reverses the dispersion before the detector integrates across the spectral dimension. Due to the dispersion, each spectral channel corresponds to a different region of the physical coded aperture
M
∈
ℝ
H
×
(
W
+
(
Λ
−
1
)
​
d
)
M\in\mathbb{R}^{H\times(W+(\Lambda-1)d)}
, where
d
d
is the pixel shift per channel. Let
Φ
i
∈
ℝ
H
×
W
\Phi_{i}\in\mathbb{R}^{H\times W}
denote the equivalent mask for the
i
i
-th channel, extracted from the corresponding region of
M
M
. The conventional forward model is
G
⁡
(
m
,
n
)
=
∑
i
=
1
Λ
Φ
i
​
(
m
,
n
)
⊙
F
i
​
(
m
,
n
)
+
ϵ
⁡
(
m
,
n
)
,
G(m,n)=\sum_{i=1}^{\Lambda}\Phi_{i}(m,n)\odot F_{i}(m,n)+\epsilon(m,n),
(1)
where
⊙
\odot
denotes element-wise multiplication and
ϵ
\epsilon
is additive noise. In matrix-vector form:
𝐠
=
𝚽
​
𝐟
+
ϵ
.
\mathbf{g}=\mathbf{\Phi}\mathbf{f}+\bm{\epsilon}.
(2)
Existing deep unfolding methods
[
13
,
14
,
15
,
16
]
adopt this PSF-free forward model, where
𝚽
T
​
𝚽
\mathbf{\Phi}^{T}\mathbf{\Phi}
is diagonal and the data fidelity subproblem admits a closed-form solution.
Figure 1:
The DD-CASSI imaging pipeline. (a) Physical optical layout consisting of two dispersive prisms, relay lenses, a coded aperture mask, and a detector. The spatially-varying, wavelength-dependent PSF is introduced by the imaging optics (indicated by the blue arrow). (b) Corresponding forward model: the input spectral cube undergoes spectral dispersion (shift), coded aperture modulation (mask), de-dispersion (de-shift), convolution with the system PSF, and spectral compression to produce the final 2D measurement. The PSF convolution step, typically ignored in existing reconstruction methods, is explicitly incorporated in our forward model.
In practice, the imaging optics of a CASSI system introduce wavelength-dependent and spatially-varying point spread function (PSF) degradation due to residual optical aberrations. As shown in Fig.
2
, the actual imaging process includes an additional PSF blurring step that is absent from the conventional model. Following the physical imaging model proposed by Song et al.
[
4
]
, we explicitly model this degradation by defining the PSF-inclusive forward operator as
G
⁡
(
m
,
n
)
=
∑
i
=
1
Λ
H
i
∗
(
Φ
i
​
(
m
,
n
)
⊙
F
i
​
(
m
,
n
)
)
+
ϵ
⁡
(
m
,
n
)
,
G(m,n)=\sum_{i=1}^{\Lambda}H_{i}*(\Phi_{i}(m,n)\odot F_{i}(m,n))+\epsilon(m,n),
(3)
The kernel
H
i
H_{i}
represents the effective end-to-end PSF of
the DD-CASSI system at wavelength
λ
i
\lambda_{i}
, obtained from
Zemax simulation of the complete optical path. Consistent
with the model in Song et al.
[
4
]
, this PSF is applied after the coded aperture modulation
in (
3
), which is a well-established
simplification in CASSI image formation.
In operator notation, the PSF-inclusive forward model becomes
𝐠
=
𝐀𝐟
+
ϵ
,
\mathbf{g}=\mathbf{A}\mathbf{f}+\bm{\epsilon},
(4)
where the forward operator
𝐀
\mathbf{A}
is defined channel-wise as
[
𝐀
⁡
(
𝐟
)
]
c
=
H
c
∗
(
Φ
c
⊙
f
c
)
.
[\mathbf{A}(\mathbf{f})]_{c}=H_{c}*(\Phi_{c}\odot f_{c}).
(5)
Note that within each reconstruction block, the PSF
H
c
H_{c}
is assumed
spatially invariant, while different blocks are assigned different
field-dependent PSFs. This piecewise treatment approximates the
continuous spatial variation of the true optical PSF at a spatial
resolution determined by the field grid density.
The corresponding adjoint operator is
[
𝐀
T
​
(
𝐫
)
]
c
=
Φ
c
⊙
(
H
c
∗
∗
𝐫
)
,
[\mathbf{A}^{T}(\mathbf{r})]_{c}=\Phi_{c}\odot(H_{c}^{*}*\mathbf{r}),
(6)
where
H
c
∗
H_{c}^{*}
denotes the conjugate of
H
c
H_{c}
, which corresponds to complex conjugation in the Fourier domain and spatial flipping in the spatial domain. Both
𝐀
\mathbf{A}
and
𝐀
T
\mathbf{A}^{T}
are implemented efficiently via FFT-based convolution.
The inclusion of the PSF fundamentally changes the structure of the inverse problem. In the conventional model,
𝚽
T
​
𝚽
\mathbf{\Phi}^{T}\mathbf{\Phi}
is diagonal, enabling closed-form inversion. In the PSF-inclusive model,
𝐀
T
​
𝐀
\mathbf{A}^{T}\mathbf{A}
involves convolutional coupling through the PSF, making it non-diagonal and precluding the closed-form solution used in existing unfolding methods. This structural difference motivates the algorithmic modifications described in Section IV.
3.2
PSF Characterization via Zernike Polynomials
The field-dependent PSF is characterized using Zernike polynomial coefficients obtained from optical design software, i.e., Zemax OpticStudio. For a DD-CASSI system with
N
f
N_{f}
field positions and
Λ
\Lambda
wavelengths, the Zernike coefficients
{
c
k
,
λ
,
j
}
\{c_{k,\lambda,j}\}
are exported for each field
k
k
, wavelength
λ
\lambda
, and Zernike term
j
j
. In this work, terms
j
=
4
,
…
,
15
j=4,\ldots,15
are used, corresponding to defocus through secondary spherical aberration; piston, tip, and tilt are excluded since they do not affect image sharpness.
Given the Zernike coefficients, the PSF for each field-wavelength combination is computed through a physical optics model:
H
k
,
λ
​
(
x
,
y
)
=
|
ℱ
−
1
​
{
P
⁡
(
u
,
v
)
⋅
e
i
​
2
​
π
λ
​
∑
j
c
k
,
λ
,
j
​
Z
j
​
(
u
,
v
)
}
|
2
H_{k,\lambda}(x,y)=\left|\mathcal{F}^{-1}\!\left\{P(u,v)\cdot e^{i\frac{2\pi}{\lambda}\sum_{j}c_{k,\lambda,j}Z_{j}(u,v)}\right\}\right|^{2}
(7)
where
P
⁡
(
u
,
v
)
P(u,v)
is the circular pupil function,
Z
j
​
(
u
,
v
)
Z_{j}(u,v)
are the Zernike basis functions, and
ℱ
−
1
\mathcal{F}^{-1}
denotes the inverse Fourier transform. This physical layer is implemented as a fixed non-learnable module that converts Zernike coefficients to PSF kernels, as shown in the upper portion of Fig.
2
.
To account for manufacturing tolerances, we perform Monte Carlo (MC) tolerance analysis in Zemax and generate
N
MC
=
1000
N_{\mathrm{MC}}=1000
realizations of the Zernike coefficients. Each realization represents a plausible manufactured lens with perturbed surface parameters. The resulting MC PSF distribution is used for data augmentation during training, as described in Section
4.5
.
