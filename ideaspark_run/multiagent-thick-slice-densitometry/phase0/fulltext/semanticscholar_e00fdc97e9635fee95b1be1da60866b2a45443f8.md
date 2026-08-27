# Efficient Unrolled Networks for Large-Scale 3D Inverse Problems

paper_id: semanticscholar:e00fdc97e9635fee95b1be1da60866b2a45443f8
tier: T2
source_used: html_arxiv
warning: none

## Intro

Linear inverse problems are ubiquitous in science and engineering, with applications ranging from medical imaging to astronomy and remote sensing. These problems typically involve recovering an unknown signal
𝒙
∗
∈
ℝ
n
{\bm{x}}^{*}\in\mathbb{R}^{n}
from noisy linear measurements
𝒚
∈
ℝ
m
{\bm{y}}\in\mathbb{R}^{m}
obtained via a known linear operator
𝑨
∈
ℝ
m
×
n
{\bm{A}}\in\mathbb{R}^{m\times n}
:
𝒚
=
𝑨
​
𝒙
∗
+
𝜺
,
{\bm{y}}={\bm{A}}{\bm{x}}^{*}+\bm{\varepsilon},
(1)
where
𝜺
\bm{\varepsilon}
represents measurement noise. Such problems are often ill-posed due to the lack of observed data, necessitating the use of regularization techniques to ensure stable and meaningful solutions.
In recent years, deep learning has emerged as a powerful tool for solving inverse problems, leveraging large datasets
69
;
31
to learn complex mappings from measurements to signals. Notable approaches include
post-processing
methods and
unrolled networks
, both of which are trained end-to-end and attempt to recover the Minimum Mean Squared Error estimator (MMSE)
41
. The former learns a direct mapping between a low-quality reconstruction (typically the adjoint or linear pseudo-inverse reconstruction) and its ground-truth reference. While straightforward and effective, it often fails to enforce data-consistency, leading to reconstructions that may not align well with the observed measurements
31
;
38
;
23
. Unrolled networks, on the other hand, integrate knowledge of the forward operator
𝑨
{\bm{A}}
into the reconstruction process by unrolling a fixed number of iterations of an optimization algorithm and replacing specific steps with learnable components
1
;
2
;
53
;
44
. They demonstrate state-of-the-art performance across a wide range of benchmarks, combining the interpretability of traditional methods with the flexibility of deep learning.
Figure 1
:
Peak video memory complexity (
dashed lines
) and global execution times (
dotted lines
) of isolated components used in unrolling. We show the cost of evaluating and back-propagating through a standard 3D data consistency step (using gradient descent) and a standard 3D network step (using a 3D DRUNet
71
). We see here that the bottleneck lies in the network step, which grows rapidly with the volume size, while the data-consistency step remains manageable even at high resolutions.
However, incorporating global operators
𝑨
{\bm{A}}
in the architecture requires training the network step at the full resolution. While in 2D problems this is typically feasible on a single GPU, the memory requirements of a global forward pass become prohibitive when working on 3D or higher dimensional problems, as illustrated in
Fig.
1
.
Deep equilibrium training reduces the memory complexity of unrolled training to that of a single pass
4
;
19
;
17
;
34
. Nonetheless, it still requires evaluating the full network at each iteration, which is infeasible on a single GPU for large-scale 3D problems.
In this work, we address this scaling challenge by introducing two complementary techniques: domain partitioning and normal operator approximation. Domain partitioning enables us to decompose a large-scale inverse problem into smaller, more manageable subproblems, allowing us to adapt the network complexity to the available resources. This approach is inspired by patch-based training methods
46
;
72
, but extends them to the context of unrolled networks for the more challenging case where the forward operator is not trivially decomposable into patches (
e.g
.
X-ray cone-beam CT). Normal operator approximation involves replacing
𝑨
⊤
​
𝑨
{\bm{A}}^{\top}{\bm{A}}
by a product of diagonal and circulant matrices
49
;
64
, enabling efficient computation of data-consistency updates via the Fast Fourier Transform (FFT). By combining these techniques, we develop a framework that allows the training and deployment of unrolled networks for arbitrarily large linear inverse problems using only one GPU. This framework can be adapted to various types of linear operators. We demonstrate the effectiveness of our approach through extensive experiments on large-scale 3D inverse problems, showcasing state-of-the-art performance while significantly reducing memory and computational requirements. Our contributions are as follows:
•
We propose a domain partitioning strategy of the operator
𝑨
{\bm{A}}
that enables the training of unrolled networks on small patches, facilitating scaling to large problems.
•
We introduce a normal operator approximation technique that leverages diagonal and circulant matrix products for efficient data-consistency updates. Notably, we show that the parameters of the factorization can be recovered efficiently by gradient descent without problem-specific data.
•
We validate our approach on large-scale 3D inverse problems, namely Multi-Coil Magnetic Resonance Imaging (MC-MRI) and Cone-Beam X-ray Computed Tomography (CBCT). We demonstrate competitive performance with reduced resource consumption,
e.g
.
handling up to
501
3
501^{3}
volumes on a single GPU.

## Method

In this section, we present a novel approach to scale unrolled networks to arbitrarily large linear inverse problems. We first introduce a
domain partitioning strategy
that allows us to reduce the size of the problem at train-time, which in turn permits the use of arbitrarily small networks that fit with memory constraints. Second, we introduce an
approximation
of the normal operator
𝑨
⊤
​
𝑨
{\bm{A}}^{\top}{\bm{A}}
as a
product of diagonal and circulant matrices
that allows to compute sub-domains in a memory efficient manner. Relying on efficient implementations of the FFT, this approximation allows a significant speed-up of the data-consistency update. A circulant matrix evaluation can be performed exactly as a diagonal product in Fourier. We first present each technique independently, then discuss how they can be combined to solve large-scale 3D inverse problems.
4.1
Domain partitioning
Consider the decomposition of
ℝ
n
\mathbb{R}^{n}
into 2 orthogonal subspaces
ℝ
n
=
ℝ
p
⊕
ℝ
q
with
​
q
=
n
−
p
.
\mathbb{R}^{n}=\mathbb{R}^{p}\oplus\mathbb{R}^{q}\hskip 10.00002pt\text{with }q=n-p.
(10)
We define the matrices
𝑺
∈
ℝ
p
×
n
{\bm{S}}\in\mathbb{R}^{p\times n}
and
𝑺
⟂
∈
ℝ
q
×
n
{\bm{S}}_{\perp}\in\mathbb{R}^{q\times n}
which extracts a vector in
ℝ
p
\mathbb{R}^{p}
, respectively in
ℝ
q
\mathbb{R}^{q}
, from
ℝ
n
\mathbb{R}^{n}
.
Using this decomposition, we assume that when solving the problem in (
1
), we already have part of the solution,
i.e
.
instead of seeking
𝒙
∗
∈
ℝ
n
{\bm{x}}^{*}\in\mathbb{R}^{n}
, we want to recover the unknown
𝒙
patch
∈
ℝ
p
{\bm{x}}_{\text{patch}}\in\mathbb{R}^{p}
such that
𝒙
∗
=
𝑺
⊤
​
𝒙
patch
+
𝑺
⟂
⊤
​
𝒙
context
,
{\bm{x}}^{*}={\bm{S}}^{\top}{\bm{x}}_{\text{patch}}+{\bm{S}}_{\perp}^{\top}{\bm{x}}_{\text{context}},
(11)
where
𝒙
context
∈
ℝ
q
{\bm{x}}_{\text{context}}\in\mathbb{R}^{q}
is known. Typically, we choose
𝒙
patch
{\bm{x}}_{\text{patch}}
as a rectangular or cuboid patch. This allows us to rewrite the linear system in (
1
) as
𝒚
~
=
𝑨
~
​
𝒙
patch
,
\tilde{{\bm{y}}}=\tilde{{\bm{A}}}{\bm{x}}_{\text{patch}},
(12)
where
𝑨
~
=
𝑨
​
𝑺
⊤
\tilde{{\bm{A}}}={\bm{A}}{\bm{S}}^{\top}
and
𝒚
~
=
𝒚
−
𝑨
​
𝑺
⟂
⊤
​
𝒙
context
\tilde{{\bm{y}}}={\bm{y}}-{\bm{A}}{\bm{S}}_{\perp}^{\top}{\bm{x}}_{\text{context}}
.
Figure 2
:
Domain partitioning strategy: in case of a forward operator
𝑨
{\bm{A}}
that mixes the signal
𝒙
{\bm{x}}
in a non-trivial manner, we can still decompose the full domain
ℝ
n
\mathbb{R}^{n}
into two orthogonal subspaces
ℝ
p
\mathbb{R}^{p}
and
ℝ
q
\mathbb{R}^{q}
. Then by linearity, we solve for the unknown smaller patch
𝒙
patch
∈
ℝ
p
{\bm{x}}_{\text{patch}}\in\mathbb{R}^{p}
(
red
) given the context
𝒙
context
∈
ℝ
q
{\bm{x}}_{\text{context}}\in\mathbb{R}^{q}
(
blue
).
We use the decomposition in (
12
) to reduce the large-scale problem into arbitrarily small ones. In the context of supervised learning, we choose
𝒙
context
=
𝑺
⟂
​
𝒙
∗
{\bm{x}}_{\text{context}}={\bm{S}}_{\perp}{\bm{x}}^{*}
.
Akin to patch-based training, we vary the position of the subspace
ℝ
p
\mathbb{R}^{p}
at random and minimize the following loss:
ℒ
PART
​
(
ϕ
)
=
𝔼
𝐒
​
𝔼
𝐱
∗
,
𝐲
​
‖
R
ϕ
⁡
(
𝐲
~
,
𝑨
~
)
−
𝐒𝐱
∗
‖
2
2
,
\displaystyle\mathcal{L}_{\text{PART}}(\phi)=\mathbb{E}_{{\mathbf{S}}}\mathbb{E}_{{\mathbf{x}}^{*},{\mathbf{y}}}~\|\operatorname{R}_{\phi}(\tilde{{\mathbf{y}}},\tilde{{\bm{A}}})-{\mathbf{S}}{\mathbf{x}}^{*}\|_{2}^{2},
(13)
with
𝑨
~
=
𝑨
𝑺
⊤
,
𝒚
~
=
𝒚
−
𝑨
𝑺
⟂
⊤
𝑺
⟂
𝒙
∗
,
\displaystyle\text{with }\tilde{{\bm{A}}}={\bm{A}}{\bm{S}}^{\top},\tilde{{\bm{y}}}={\bm{y}}-{\bm{A}}{\bm{S}}_{\perp}^{\top}{\bm{S}}_{\perp}{\bm{x}}^{*},
where
𝑺
​
𝒙
∗
∈
ℝ
p
{\bm{S}}{\bm{x}}^{*}\in\mathbb{R}^{p}
is the ground-truth patch corresponding to the subspace
ℝ
p
\mathbb{R}^{p}
.
Test-time without ground-truth context
We deploy the network
R
ϕ
\operatorname{R}_{\phi}
, trained with
domain partitioning
, in a two-step procedure:
1.
At test-time, we can mitigate the complexity of evaluating the prior step
D
ϕ
\operatorname{D}_{\phi}
on the full volume by evaluating it sequentially along patches and merging the processed patches. Thus, we obtain an estimation
𝒙
~
\tilde{{\bm{x}}}
of the ground-truth signal by solving the full problem in (
1
). We use a standard unrolled scheme, denoted by
𝒙
~
=
R
ϕ
⁡
(
𝒚
,
𝑨
)
\tilde{{\bm{x}}}=\operatorname{R}_{\phi}({\bm{y}},{\bm{A}})
, where the data-step is computed on the whole volume,
i.e
.
without partitioning, and the prior step
D
ϕ
\operatorname{D}_{\phi}
is performed patch-by-patch.
2.
We use the estimation
𝒙
~
\tilde{{\bm{x}}}
to build a context within our
domain-partitioned
framework,
i.e
.
𝒙
context
=
𝑺
⟂
​
𝒙
~
{\bm{x}}_{\text{context}}={\bm{S}}_{\perp}\tilde{{\bm{x}}}
, and independently solve each subproblem using the same network
R
ϕ
\operatorname{R}_{\phi}
.
For the CBCT experiments, we observe that this two-step refining procedure consistently improves the reconstruction quality compared to using only the initial estimation
𝒙
~
\tilde{{\bm{x}}}
. For the MC-MRI experiments, we observe that the first estimation
𝒙
~
\tilde{{\bm{x}}}
is already of high-quality, thus the second refinement step brings negligible improvements. A summary of the test-time algorithm is provided in
Algorithm
1
, and we give more details in
Appendix
B
.
init:
𝒚
∈
ℝ
m
{\bm{y}}\in\mathbb{R}^{m}
,
𝑨
∈
ℝ
m
×
n
{\bm{A}}\in\mathbb{R}^{m\times n}
,
𝒙
0
=
𝑨
†
​
𝒚
∈
ℝ
n
{\bm{x}}_{0}={\bm{A}}^{\dagger}{\bm{y}}\in\mathbb{R}^{n}
,
k
=
0
,
K
>
0
k=0,K>0
, step size
η
>
0
\eta>0
,
R
ϕ
\operatorname{R}_{\phi}
.
First unrolling procedure to get
𝐱
~
≈
𝐱
∗
\tilde{{\bm{x}}}\approx{\bm{x}}^{*}
𝒙
~
←
R
ϕ
⁡
(
𝒚
,
𝑨
)
\tilde{{\bm{x}}}\leftarrow\operatorname{R}_{\phi}({\bm{y}},{\bm{A}})
Domain partitioned computation
X
patches
=
[
]
X_{\mathrm{patches}}=[~]
for
𝐒
{\bm{S}}
in
patches(
)
do
init:
𝒙
context
=
𝑺
⟂
​
𝒙
~
{\bm{x}}_{\text{context}}={\bm{S}}_{\perp}\tilde{{\bm{x}}}
,
𝑨
~
=
𝑨
​
𝑺
⊤
\tilde{{\bm{A}}}={\bm{A}}{\bm{S}}^{\top}
,
𝒚
~
=
𝒚
−
𝑨
​
𝑺
⟂
⊤
​
𝑺
⟂
​
𝒙
~
\tilde{{\bm{y}}}={\bm{y}}-{\bm{A}}{\bm{S}}_{\perp}^{\top}{\bm{S}}_{\perp}\tilde{{\bm{x}}}
(
12
)
𝒙
^
patch
←
R
ϕ
⁡
(
𝒚
~
,
𝑨
~
)
\hat{{\bm{x}}}_{\text{patch}}\leftarrow\operatorname{R}_{\phi}(\tilde{{\bm{y}}},\tilde{{\bm{A}}})
append(
X
patches
X_{\mathrm{patches}}
,
𝐱
^
patch
\hat{{\bm{x}}}_{\text{patch}}
)
𝒙
^
←
\hat{{\bm{x}}}\leftarrow
aggregate(
X
patches
X_{\mathrm{patches}}
)
Algorithm 1
Test-time domain partitioned inference
4.2
Normal operator approximation
We previously introduced the domain partitioning decomposition where
𝒚
~
=
𝑨
~
​
𝒙
patch
\tilde{{\bm{y}}}=\tilde{{\bm{A}}}{\bm{x}}_{\text{patch}}
with
𝑨
~
=
𝑨
​
𝑺
⊤
\tilde{{\bm{A}}}={\bm{A}}{\bm{S}}^{\top}
. Note here that solving (
12
) still requires the global forward operator and its adjoint,
i.e
.
through the evaluation of
𝑨
~
⊤
​
𝑨
~
=
𝑺
​
𝑨
⊤
​
𝑨
​
𝑺
⊤
\tilde{{\bm{A}}}^{\top}\tilde{{\bm{A}}}={\bm{S}}{\bm{A}}^{\top}{\bm{A}}{\bm{S}}^{\top}
. In this section, we introduce an efficient approximation of the normal operator
𝑨
⊤
​
𝑨
{\bm{A}}^{\top}{\bm{A}}
that significantly reduces the computational burden of naively evaluating
𝑨
~
⊤
​
𝑨
~
\tilde{{\bm{A}}}^{\top}\tilde{{\bm{A}}}
.
Let us recall the data-consistency update typically computed in the form of a gradient descent step:
h
⁡
(
𝒙
)
\displaystyle h({\bm{x}})
=
𝒙
−
η
​
∇
𝒙
d
​
(
𝑨
​
𝒙
,
𝒚
)
\displaystyle={\bm{x}}-\eta\nabla_{{\bm{x}}}d({\bm{A}}{\bm{x}},{\bm{y}})
(14)
=
𝒙
−
η
⁡
(
𝑨
⊤
​
𝑨
)
​
𝒙
+
𝑨
⊤
​
𝒚
.
\displaystyle={\bm{x}}-\eta({\bm{A}}^{\top}{\bm{A}}){\bm{x}}+{\bm{A}}^{\top}{\bm{y}}.
Discarding the constant term
𝑨
⊤
​
𝒚
{\bm{A}}^{\top}{\bm{y}}
which can be pre-computed, we see that a computing (
14
) only requires the evaluation of the normal operator
𝑨
⊤
​
𝑨
​
𝒙
{\bm{A}}^{\top}{\bm{A}}{\bm{x}}
.
Remark 4.1
Focusing on the evaluation of the normal operator
𝐀
⊤
​
𝐀
{\bm{A}}^{\top}{\bm{A}}
is not restrictive. In the case of the data consistency step being a proximal step,
i.e
.
h
(
x
)
=
prox
η
d
(
𝐀
⋅
,
𝐲
)
(
𝐱
)
h(x)=\mathrm{prox}_{\eta d({\bm{A}}\cdot,{\bm{y}})}({\bm{x}})
,
𝐀
⊤
​
𝐀
{\bm{A}}^{\top}{\bm{A}}
is again the main component in most popular solvers,
e.g
.
conjugate gradient.
Translation-equivariant operators
Assuming that
𝑨
{\bm{A}}
is translation-equivariant, then
𝑨
⊤
​
𝑨
{\bm{A}}^{\top}{\bm{A}}
is a convolutional operator
64
. In this case, we can leverage the convolution theorem to efficiently compute the normal operator evaluation in the Fourier domain. More precisely, we can write
𝑨
⊤
​
𝑨
​
𝒙
=
𝑭
−
1
​
diag
​
(
𝝀
)
​
𝑭
​
𝒙
,
{\bm{A}}^{\top}{\bm{A}}{\bm{x}}={\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}}{\bm{x}},
(15)
where
𝑭
{\bm{F}}
and
𝑭
−
1
{\bm{F}}^{-1}
are the Fourier and inverse Fourier transforms, respectively, and
𝝀
∈
ℂ
n
\bm{\lambda}\in{\mathbb{C}}^{n}
is the frequency response of the convolution kernel associated with
𝑨
⊤
​
𝑨
{\bm{A}}^{\top}{\bm{A}}
.
Spatial modulation
We can generalize the factorization to a larger class of non-translation equivariant operators,
e.g
.
inpainting, by modulating the output of the convolution by a diagonal operation in the spatial domain. More precisely, we factorize the normal operator as
𝑨
⊤
​
𝑨
=
𝑯
=
diag
​
(
𝒎
)
​
𝑭
−
1
​
diag
​
(
𝝀
)
​
𝑭
,
{\bm{A}}^{\top}{\bm{A}}={\bm{H}}={\textnormal{diag}}({\bm{m}}){\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}},
(16)
where
𝒎
∈
ℝ
n
{\bm{m}}\in\mathbb{R}^{n}
is homogeneous to a spatial sensitivity map, or mask.
Following Schmid
et al
.
49
, we could increase the expressivity of the approximation by adding more diagonal-circulant factors,
i.e
.
𝑯
=
∏
i
=
1
N
diag
​
(
𝒎
i
)
​
𝑭
−
1
​
diag
​
(
𝝀
i
)
​
𝑭
{\bm{H}}=\prod_{i=1}^{N}{\textnormal{diag}}({\bm{m}}_{i}){\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}_{i}){\bm{F}}
. However, in practice we observe that a single factorization (
N
=
1
N=1
) is sufficient to obtain good reconstruction results while keeping the computational cost low.
Remark 4.2
Note that in (
16
) we approximate the symmetric operator
𝐀
⊤
​
𝐀
{\bm{A}}^{\top}{\bm{A}}
with a non-symmetric factorization
𝐇
{\bm{H}}
. In practice, we observe that imposing symmetry constraints during the fitting procedure does not improve the quality of the approximation nor the final reconstruction results. At test-time, using either
𝐇
{\bm{H}}
or its symmetrized version
1
2
​
(
𝐇
+
𝐇
⊤
)
\frac{1}{2}({\bm{H}}+{\bm{H}}^{\top})
yields similar results, thus we keep the simpler version
𝐇
{\bm{H}}
for efficiency.
Examples
For CT, the Fourier slice theorem states that each row of the forward operator corresponds to sampling a
radial
line in the Fourier domain
16
, which is exactly written as a diagonal operation in frequency space. The spatial modulation
diag
​
(
𝒎
)
{\textnormal{diag}}({\bm{m}})
can then be interpreted as a resampling map which compensates the
cartesian
sampling done by the FFT, rather than a
radial
computation. For other modalities, such as deconvolution problems,
𝑨
{\bm{A}}
is exactly a convolutional operator and
𝒎
{\bm{m}}
is simply an all-one vector. For inpainting problems,
𝑨
{\bm{A}}
is a diagonal binary masking operator and
𝑨
⊤
​
𝑨
=
𝑨
{\bm{A}}^{\top}{\bm{A}}={\bm{A}}
, which can be exactly represented by our proposed form in (
16
) with
𝝀
\bm{\lambda}
being an all-one vector.
Remark 4.3
For multi-coil MRI, the normal operator writes
𝐀
⊤
​
𝐀
=
∑
c
=
1
C
𝐒
c
⊤
​
𝐅
−
1
​
𝐌
⊤
​
𝐌
​
𝐅
​
𝐒
c
{\bm{A}}^{\top}{\bm{A}}=\sum_{c=1}^{C}{\bm{S}}_{c}^{\top}{\bm{F}}^{-1}{\bm{M}}^{\top}{\bm{M}}{\bm{F}}{\bm{S}}_{c}
, where
𝐒
c
{\bm{S}}_{c}
is the sensitivity map of coil
c
c
and
𝐌
{\bm{M}}
is the undersampling mask in frequency space. In this case, the normal operator cannot be exactly represented by our proposed form in (
16
). Indeed, the approximation breaks down to finding a single-coil equivalent of a multi-coil forward model. In practice, we observe that by slightly changing the factorization to
𝐇
=
diag
​
(
𝐦
)
H
​
𝐅
−
1
​
diag
​
(
𝛌
)
​
𝐅
​
diag
​
(
𝐦
)
{\bm{H}}={\textnormal{diag}}({\bm{m}})^{H}{\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}}{\textnormal{diag}}({\bm{m}})
, we can obtain a good approximation of the multi-coil normal operator.
Fitting the approximation
We fit the parameters
𝒎
{\bm{m}}
and
𝝀
\bm{\lambda}
by gradient descent on a set of random vectors
{
𝒙
i
}
i
=
1
N
\{{\bm{x}}_{i}\}_{i=1}^{N}
. More precisely, we minimize the following loss:
ℒ
⁡
(
𝒎
,
𝝀
)
=
𝔼
𝐱
∼
𝒩
⁡
(
𝟎
,
𝑰
)
​
‖
𝑨
⊤
​
𝑨
​
𝐱
−
𝑯
⁡
(
𝒎
,
𝝀
)
​
𝐱
‖
2
2
.
\mathcal{L}({\bm{m}},\bm{\lambda})=\mathbb{E}_{{\mathbf{x}}\sim\mathcal{N}(\bm{0},{\bm{I}})}\|{\bm{A}}^{\top}{\bm{A}}{\mathbf{x}}-{\bm{H}}({\bm{m}},\bm{\lambda}){\mathbf{x}}\|_{2}^{2}.
(17)
Using properties of standard Gaussian random vectors, note that (
17
) is equal to the Frobenius norm of the residual,
i.e
.
ℒ
⁡
(
𝒎
,
𝝀
)
\displaystyle\mathcal{L}({\bm{m}},\bm{\lambda})
=
𝔼
𝐱
∼
𝒩
⁡
(
𝟎
,
𝑰
)
​
‖
𝑨
⊤
​
𝑨
​
𝐱
−
𝑯
⁡
(
𝒎
,
𝝀
)
​
𝐱
‖
2
2
\displaystyle=\mathbb{E}_{{\mathbf{x}}\sim\mathcal{N}(\bm{0},{\bm{I}})}\|{\bm{A}}^{\top}{\bm{A}}{\mathbf{x}}-{\bm{H}}({\bm{m}},\bm{\lambda}){\mathbf{x}}\|_{2}^{2}
(18)
=
‖
𝑨
⊤
​
𝑨
−
𝑯
⁡
(
𝒎
,
𝝀
)
‖
F
2
\displaystyle=\|{\bm{A}}^{\top}{\bm{A}}-{\bm{H}}({\bm{m}},\bm{\lambda})\|_{F}^{2}
This means that we can fit the parameters
(
𝒎
,
𝝀
)
({\bm{m}},\bm{\lambda})
of the factorization without any problem-specific data, which is particularly useful when the forward operator
𝑨
{\bm{A}}
is known but a limited amount of training data is available.
4.3
Efficient approximation on partitioned domain
When using the domain partitioning strategy presented in
Section
4.1
, we need to efficiently compute the normal operator evaluation on any patch
𝒙
patch
∈
ℝ
p
{\bm{x}}_{\text{patch}}\in\mathbb{R}^{p}
,
i.e
.
𝑨
~
⊤
​
𝑨
~
​
𝒙
patch
=
𝑺
​
𝑨
⊤
​
𝑨
​
𝑺
⊤
​
𝒙
patch
\tilde{{\bm{A}}}^{\top}\tilde{{\bm{A}}}{\bm{x}}_{\text{patch}}={\bm{S}}{\bm{A}}^{\top}{\bm{A}}{\bm{S}}^{\top}{\bm{x}}_{\text{patch}}
. Using the approximation in (
16
), we can write
𝑨
~
⊤
​
𝑨
~
​
𝒙
patch
≈
𝑺
​
diag
​
(
𝒎
)
​
𝑭
−
1
​
diag
​
(
𝝀
)
​
𝑭
​
𝑺
⊤
​
𝒙
patch
.
\tilde{{\bm{A}}}^{\top}\tilde{{\bm{A}}}{\bm{x}}_{\text{patch}}\approx{\bm{S}}{\textnormal{diag}}({\bm{m}}){\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}}{\bm{S}}^{\top}{\bm{x}}_{\text{patch}}.
(19)
Let us decompose (
19
) into finer steps:
(i)
𝑺
⊤
​
𝒙
patch
{\bm{S}}^{\top}{\bm{x}}_{\text{patch}}
is a zero-padding operation, that materializes in memory a vector of size
n
n
, from a patch of size
p
≪
n
p\ll n
,
(ii)
𝑭
−
1
​
diag
​
(
𝝀
)
​
𝑭
{\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}}
is equivalent to a convolution with a kernel of size
n
n
,
(iii)
𝑺
​
diag
​
(
𝒎
)
{\bm{S}}{\textnormal{diag}}({\bm{m}})
is a diagonal operator of size
p
p
.
Here, note that both steps
(i)
and
(ii)
are highly inefficient as they require going back to the full signal volume. As the input of the convolution is a zero-padded cuboid, we can restrict the convolution kernel to a smaller size
k
=
2
​
p
≪
n
k=2p\ll n
and maintain exact computation. More precisely, we can write
𝑺
​
diag
​
(
𝒎
)
​
𝑭
−
1
​
diag
​
(
𝝀
)
​
𝑭
​
𝑺
⊤
​
𝒙
patch
\displaystyle{\displaystyle\bm{S}}{\textnormal{diag}}({\bm{m}}){\bm{F}}^{-1}{\textnormal{diag}}(\bm{\lambda}){\bm{F}}{\bm{S}}^{\top}{\bm{x}}_{\text{patch}}
(20)
=
\displaystyle=
diag
​
(
𝑺
​
𝒎
)
​
𝑭
k
−
1
​
diag
​
(
𝝀
k
)
​
𝑭
k
​
𝒙
patch
,
\displaystyle{\displaystyle\textnormal{diag}}({\bm{S}}{\bm{m}}){\bm{F}}_{k}^{-1}{\textnormal{diag}}(\bm{\lambda}_{k}){\bm{F}}_{k}{\bm{x}}_{\text{patch}},
where
𝑭
k
{\bm{F}}_{k}
and
𝑭
k
−
1
{\bm{F}}_{k}^{-1}
are the Fourier and inverse Fourier transforms restricted to size
k
k
(with zero-padding or cropping from
ℝ
p
\mathbb{R}^{p}
to
ℝ
k
\mathbb{R}^{k}
), respectively, and
𝝀
k
∈
ℂ
k
\bm{\lambda}_{k}\in{\mathbb{C}}^{k}
is the frequency response of the convolution kernel restricted to size
k
k
.
Figure 3
:
Illustrations of sparse view reconstructions with [30/1200] projections on the Walnut-CBCT
10
dataset using the methods compared in Tab.
1
.
First row
axial slices,
second row
vertical slices from the same sample. PSNR is computed per slice.
