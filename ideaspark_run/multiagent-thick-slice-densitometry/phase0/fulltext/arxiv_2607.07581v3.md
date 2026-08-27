# Cardiac MRI Through-Plane Super-Resolution Guided by Reference and Memory

paper_id: arxiv:2607.07581v3
tier: T3
source_used: html_arxiv
warning: none

## Intro

Cardiac MRI (CMR) relies on 2D breath-hold acquisitions with high in-plane resolution but coarse through-plane resolution, limiting downstream 3D analysis and diagnosis
[
21
]
. Standard CMR protocols routinely acquire a small set of long-axis views as complementary appearance cues for the short-axis stack or acquire orthogonal anisotropic 2D image stacks for 3D interpretation from different viewpoints.
Recovering an isotropic 3D volume from such anisotropic stacks is commonly framed as slice-to-volume reconstruction (SVR)
[
12
]
, which proceeds in two sequential steps: (i)
slice alignment
, which corrects rigid inter-breath-hold misalignment between 2D image stacks via slice-to-slice or slice-to-volume registration
[
23
,
1
]
, and (ii)
through-plane super-resolution (SR)
, which fuses the aligned anisotropic data into a coherent high-resolution (HR) 3D volume. The registration step has been extensively studied, with mature solutions based on segmentation map intersection matching
[
20
,
2
,
24
]
. In this work, we focus on the second step and assume the input low-resolution (LR) stacks are already spatially aligned, aiming to learn an accurate through-plane SR model that recovers fine anatomical details using cross-view HR images as guidance.
Early model-based methods formulate SR as an inverse problem regularized by handcrafted priors
[
19
,
9
,
18
,
21
]
, but rely on linear degradation assumptions and struggle to recover fine details at large upsampling factors. Deep learning has since become the dominant paradigm for MRI SR
[
11
]
, with supervised methods learning LR-to-HR mapping via densely connected CNNs
[
4
]
, multiscale convolutional networks
[
17
,
6
]
, and transformers
[
13
]
. However, most existing approaches operate on single-input or implicitly fuse multi-input data, limiting their ability to exploit complementary structural information from auxiliary views, and typically process slices independently, leading to inter-slice discontinuities in the reconstructed volume. Self-supervised methods such as SMORE
[
27
]
and SSGNN
[
22
]
avoid external training data but similarly lack mechanisms for cross-view correspondence.
We therefore propose a reference- and memory-based method,
STRMSR
, for the SR step of cardiac MRI SVR. Given a target LR volume as a video, we leverage auxiliary HR reference views from the same subject to guide single-frame through-plane SR and take the intermediate SR outputs as the memory to propagate SR results along the third axis. Our contributions are summarized as follows:
(1) We propose coarse-to-fine correspondence matching and patch-wise dynamic feature aggregation to enable robust detail transfer between target and reference/memory images under cross-view geometric misalignment.
(2) We introduce a memory-based inter-slice SR propagation mechanism inspired by recent advances in video object segmentation
[
16
,
5
,
25
]
, which enforces slice-to-slice consistency and improves volumetric coherence in the reconstructed 3D HR output.
(3) We validate the proposed method on the WHS cardiac MRI dataset under two reference protocols to demonstrate its effectiveness across distinct geometries and reference coverage densities.

## Method

Figure 1:
(a) Overall architecture of STRMSR, where we perform coarse-to-fine contextual matching (CFCM) and (b) patch-wise dynamic feature aggregation (PDFA). (c) The intermediate super-resolution results are stored in the memory bank.
2.1
Dual-Branch Transformer for Feature Extraction
We first downsample the reference (or memory) HR image (
I
H
​
R
I_{HR}
) into the same through-plane resolution as the target LR image (
I
T
I_{T}
), denoting as
I
R
I_{R}
.
As shown in Fig.
1
(a), we then extract target and reference features (
F
T
F_{T}
and
F
R
F_{R}
) using a dual-branch encoder based on Swin Transformer group (STG), consisting of Residual Swin Transformer Blocks (RSTB). Target LR and reference LR branches share parameters to keep their embeddings in a common feature space, while the reference HR images are encoded by an independent branch to preserve resolution-specific details.
For LR inputs (
I
T
I_{T}
and
I
R
I_{R}
), we first apply center-copy upsampling along the through-plane axis to match the HR reference resolution, then extract features
F
T
/
R
s
F_{T/R}^{s}
at finer levels (
s
=
2
,
3
s=2,3
) using strided convolutions (
s
​
t
​
r
​
i
​
d
​
e
=
2
stride=2
). Lastly, we apply RSTB only at the coarsest level (
s
=
1
s=1
):
F
T
/
R
1
=
RSTB
LR
​
(
F
T
/
R
2
)
+
F
T
/
R
2
.
F_{T/R}^{1}=\mathrm{RSTB}_{\mathrm{LR}}(F_{T/R}^{2})+F_{T/R}^{2}.
(1)
For HR reference (
I
H
​
R
I_{HR}
), we encode features
F
H
​
R
F_{HR}
at all levels using RSTB blocks:
F
H
​
R
s
=
RSTB
HR
s
​
(
F
H
​
R
s
+
1
)
+
F
H
​
R
s
+
1
,
s
∈
{
1
,
2
,
3
}
,
F
H
​
R
4
=
I
H
​
R
.
F_{HR}^{s}=\mathrm{RSTB}_{\mathrm{HR}}^{s}(F_{HR}^{s+1})+F_{HR}^{s+1},\quad s\in\{1,2,3\},\quad F_{HR}^{4}=I_{HR}.
(2)
2.2
Coarse-to-Fine Contextual Matching (CFCM)
Figure 2:
Illustration of CFCM at the coarsest level
(
s
=
1
s{=}1
).
Given multi-scale feature pyramids
{
F
T
s
,
F
R
s
,
F
H
​
R
s
}
s
∈
{
1
,
2
,
3
}
\{F_{T}^{s},F_{R}^{s},F_{HR}^{s}\}_{s\in\{1,2,3\}}
at resolutions
H
/
4
H/4
,
H
/
2
H/2
, and
H
H
,
we aim to build dense correspondence between
F
T
s
F_{T}^{s}
and
F
R
s
F_{R}^{s}
, then warp the HR features
F
H
​
R
s
F_{HR}^{s}
accordingly. To ensure spatial consistency during center propagation across levels, a strict pyramid is enforced by padding the coarsest level when required and defining finer levels as exact
2
×
2\times
and
4
×
4\times
upscalings.
As illustrated in Fig.
2
, at the coarsest level (
s
=
1
s=1
), we partition
F
T
1
F_{T}^{1}
into
M
M
non-overlapping
k
×
k
k\times k
blocks (
ℬ
\mathcal{B}
). For each target block
i
i
in
F
T
1
F_{T}^{1}
, we compute multi-dilation normalized correlation
[
15
]
with all blocks in the reference feature
F
R
1
F_{R}^{1}
to locate its best match at block
j
j
. We then crop a local search region
ℛ
\mathcal{R}
of size
d
×
d
d\times d
centered at
i
i
and
j
j
.
Within each local search region pair, we perform dense
3
×
3
3\times 3
patch matching. For each target patch
p
u
p_{u}
and reference patch
q
v
q_{v}
, we compute cosine similarity
[
13
,
15
]
and obtain correspondence:
𝒮
u
​
v
=
⟨
p
u
/
‖
p
u
‖
,
q
v
/
‖
q
v
‖
⟩
,
ℐ
u
=
arg
⁡
max
v
⁡
𝒮
u
​
v
,
𝒲
u
=
max
v
⁡
𝒮
u
​
v
,
\mathcal{S}_{uv}=\langle p_{u}/\|p_{u}\|,\,q_{v}/\|q_{v}\|\rangle,\quad\mathcal{I}_{u}=\arg\max_{v}\mathcal{S}_{uv},\quad\mathcal{W}_{u}=\max_{v}\mathcal{S}_{uv},
(3)
where
ℐ
\mathcal{I}
is the index map indicating the best matching position, and
𝒲
\mathcal{W}
is the confidence map.
We extract corresponding
3
×
3
3\times 3
patches from the reference HR feature
F
H
​
R
s
F_{HR}^{s}
via
ℐ
\mathcal{I}
and assemble them via confidence-weighted folding:
F
W
1
=
fold
⁡
(
{
𝒲
u
⋅
F
H
​
R
1
​
[
ℐ
u
]
}
i
)
,
F_{W}^{1}=\mathrm{fold}\left(\{\mathcal{W}_{u}\cdot F_{HR}^{1}[\mathcal{I}_{u}]\}_{i}\right),
(4)
where
[
⋅
]
[\cdot]
denotes patch extraction,
fold
\mathrm{fold}
performs the inverse of unfolding with overlap averaging, and
F
W
1
F_{W}^{1}
denotes the warped HR reference feature.
Different from McMRSR
[
13
]
, which mainly reuses the LR-level correspondence for multi-scale feature transfer, our CFCM uses it only as an initialization and further refines the correspondence at finer levels
s
∈
2
,
3
s\in{2,3}
. The matched block centers
𝒞
i
/
j
\mathcal{C}_{i/j}
propagate from level
s
s
to
s
+
1
s+1
via:
𝒞
i
/
j
s
+
1
=
2
⋅
NN
⁡
(
𝒞
i
/
j
s
)
,
\mathcal{C}_{i/j}^{s+1}=2\cdot\mathrm{NN}(\mathcal{C}_{i/j}^{s}),
(5)
where nearest-neighbor interpolation (
NN
\mathrm{NN}
) preserves block boundaries. At each finer level, we perform a new dense local search within radius
r
r
around the propagated center to update the correspondence, producing warped HR reference features
F
W
2
F_{W}^{2}
and
F
W
3
F_{W}^{3}
.
2.3
Patch-wise Dynamic Feature Aggregation (PDFA)
Given
Z
Z
warped feature maps
{
F
W
,
z
s
}
z
=
1
Z
\{F_{W,z}^{s}\}_{z=1}^{Z}
at scale
s
∈
{
1
,
2
,
3
}
s\in\{1,2,3\}
obtained from
Z
Z
multiple HR references and propagated memory frames, the goal is to fuse them into a single spatially aligned feature
F
A
s
F_{A}^{s}
. Direct pixel-wise gating is expensive and tends to overfit to local misalignments.
We therefore propose a patch-wise dynamic aggregation strategy, as shown in Fig.
1
(b).
Formally, we split each
F
W
,
z
s
∈
ℝ
C
×
H
×
W
F_{W,z}^{s}\in\mathbb{R}^{C\times H\times W}
into non-overlapping
p
×
p
p\times p
patches indexed by
h
h
, and compute a patch descriptor via average pooling:
d
z
,
h
=
AvgPool
⁡
(
F
W
,
z
,
h
s
)
∈
ℝ
C
.
d_{z,h}=\mathrm{AvgPool}(F_{W,z,h}^{s})\in\mathbb{R}^{C}.
(6)
A lightweight MLP
g
⁡
(
⋅
)
g(\cdot)
then produces a patch logit
ℓ
z
,
h
=
g
⁡
(
d
z
,
h
)
\ell_{z,h}=g(d_{z,h})
. Finally, we use softmax to normalize logits across the
Z
Z
candidates and broadcast weights to pixel resolution within the patch, yielding the fused feature:
α
z
,
h
=
exp
⁡
(
ℓ
z
,
h
)
∑
j
=
1
Z
exp
⁡
(
ℓ
j
,
h
)
,
F
A
s
​
(
x
,
y
)
=
∑
z
=
1
Z
α
z
,
h
⁡
(
x
,
y
)
​
F
W
,
z
s
​
(
x
,
y
)
,
\alpha_{z,h}=\frac{\exp(\ell_{z,h})}{\sum_{j=1}^{Z}\exp(\ell_{j,h})},\qquad F_{A}^{s}(x,y)=\sum_{z=1}^{Z}\alpha_{z,h(x,y)}\,F_{W,z}^{s}(x,y),
(7)
where
h
⁡
(
x
,
y
)
h(x,y)
denotes the patch index containing pixel
(
x
,
y
)
(x,y)
. This formulation can be interpreted as a content-adaptive mixture-of-experts
[
3
,
26
]
over warped references/memories, enabling selective transfer of informative patches while suppressing inconsistent matches.
2.4
Progressive Multi-Scale Feature Fusion and Reconstruction
Given fused HR reference/memory features
{
F
A
s
}
s
∈
{
1
,
2
,
3
}
\{F_{A}^{s}\}_{s\in\{1,2,3\}}
aggregated from multiple reference/memory frames, we reconstruct the target HR image by combining feature alignment and detail refinement via Feature Alignment Module (FAM) and Dual-Path Refinement Block (DPRB) as in
[
13
]
.
2.5
Memory-based SR Propagation
We take the LR volume as a video, where the third axis is deemed as the temporal axis. As shown in Fig.
1
(c), once we use the reference images to super resolve the first frame, we store the
I
T
I_{T}
-
I
S
​
R
I_{SR}
pair in a First-In-First-Out (FIFO) memory bank. The reference images together with memory images can then guide SR of following frames, where the results can be further stored in the memory bank to improve slice-to-slice SR consistency.
2.6
Loss Function
We optimize the network using a combination of an image-domain
ℓ
1
\ell_{1}
reconstruction loss and a
k
k
-space loss:
ℒ
=
λ
rec
​
‖
I
S
​
R
−
I
G
​
T
‖
1
+
λ
k
​
‖
ℱ
⁡
(
I
S
​
R
)
−
ℱ
⁡
(
I
G
​
T
)
‖
2
2
\mathcal{L}=\lambda_{\mathrm{rec}}\left\|I_{SR}-I_{GT}\right\|_{1}+\lambda_{\mathrm{k}}\left\|\mathcal{F}(I_{SR})-\mathcal{F}(I_{GT})\right\|_{2}^{2}
,
where
I
G
​
T
I_{GT}
is the ground truth HR image,
ℱ
⁡
(
⋅
)
\mathcal{F}(\cdot)
denotes the 2D FFT applied slice-wise, and we set
λ
rec
=
1
\lambda_{\mathrm{rec}}=1
and
λ
k
=
0.001
\lambda_{\mathrm{k}}=0.001
in all experiments.
