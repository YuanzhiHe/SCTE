# Solving Inverse Problems with Model Mismatch using Untrained Neural Networks within Model-based Architectures

paper_id: arxiv:2403.04847v2
tier: H
source_used: html_arxiv
warning: none

## Intro

Consider an inverse problem of the following form:
𝒚
=
𝒜
⁡
(
𝒙
)
+
ϵ
.
\bm{y}=\mathcal{A}(\bm{x})+\bm{\epsilon}.
(1)
The goal is to reconstruct the latent signal
𝒙
\bm{x}
from the measurements
𝒚
\bm{y}
in the presence of noise
ϵ
\bm{\epsilon}
, where typically the forward model
𝒜
\mathcal{A}
is assumed to be known. Inverse problems are generally challenging because they are often ill-posed,
i.e.,
the solution is not unique, or the reconstruction is highly sensitive to noise and/or model mismatch.
The traditional approach to recovering
𝒙
\bm{x}
from the measurements
𝒚
\bm{y}
is by solving a regularized optimization problem of the form:
min
𝒙
⁡
1
2
​
‖
𝒚
−
𝒜
⁡
(
𝒙
)
‖
2
2
+
γ
​
r
​
(
𝒙
)
,
\min_{\bm{x}}~\frac{1}{2}~\|\bm{y}-\mathcal{A}(\bm{x})\|^{2}_{2}+\gamma r(\bm{x}),
(2)
where
γ
≥
0
\gamma\geq 0
is an appropriately-chosen parameter. The regularizer
r
r
is usually predetermined based on some known or desired structure, e.g.,
ℓ
1
\ell_{1}
-,
ℓ
2
\ell_{2}
-, or total variation (TV) norm to promote sparsity, smoothness, or edges in image reconstructions, respectively. Solving (
2
) requires careful consideration of the underlying physics or the forward model
𝒜
\mathcal{A}
to obtain a stable and accurate reconstruction. However, knowing
𝒜
\mathcal{A}
can be challenging in practice. The reasons for this include inaccurate measurements and challenging calibration, highly nonlinear and/or computationally expensive models replaced by simplified versions, as well as access to only approximations of certain features.
Usually, some knowledge of the true model, designated
𝒜
0
\mathcal{A}_{0}
in this work, is available. This occurs in many applications, including blind deconvolution/deblurring problems, recovering seismic layer models using the simplified acoustic wave equation as the forward model
(
Mousa & Al-Shuhail 2011
)
, determining fault locations in media with unknown structures
(
Mahmoud & Khalid 2013
)
, computational tomography (CT) when the dynamic behavior of a human object is observed during the measurement period
Blanke et al. 2020
, sparse signal recovery when the sensing matrix is not known perfectly, etc.
When the forward model is precisely known, the inverse problem in (
2
) can be solved via classical optimization methods, where
r
r
is predefined. Machine-learning approaches, as summarized in
Arridge et al. 2019
;
Iqbal et al. 2023
;
Ongie et al. 2020
, have demonstrated superior reconstruction performance by effectively learning a regularizer from data. For example, the Plug-and-Play method
(
Venkatakrishnan et al. 2013
)
trains a general denoiser independently of the forward model and iteratively minimizes (
2
) with the learned denoiser as the regularization updates. Another class of approaches, the
loop unrolling
(LU) method
(
Gregor & LeCun 2010
;
Hershey et al. 2014
;
Adler & Öktem 2018
)
and its extension using deep equilibrium models
(
Gilton et al. 2021a
)
, build on the observation that many iterative algorithms for solving (
2
) can be re-expressed as a sequence of neural networks
(
Gregor & LeCun 2010
)
, which can then be trained to remove artifacts and noise patterns associated with a known
𝒜
\mathcal{A}
, resulting in higher quality reconstructions.
Figure 1
:
A proximal LU network is trained for a deblurring task using a
single
forward model. The top row shows the intermediate reconstructions over 8 iterations using the true model, while the bottom row shows the evaluation results when a small perturbation is added to the forward model (the Peak Signal-to-Noise Ratio of the true kernel to the noisy kernel is 40.9 dB). This quality degradation is due to the accumulation of errors in the forward model.
While these model-based machine learning solvers have demonstrated impressive performance, they encounter challenges while dealing with inaccurate forward models. Plug-and-Play, LU, and its deep equilibrium extensions all entail a gradient update of the data-fidelity term, which can be in error when the forward model is inaccurate. This causes the learned denoiser or regularizer to be ineffective. In particular, while LU often exhibits state-of-the-art performance, has faster runtime, and has improved stability in training
(
Guan et al. 2022
)
compared to other approaches, it requires precise knowledge of the forward model. Fig.
1
demonstrates the sensitivity of LU to mismatch in
𝒜
\mathcal{A}
.
Some works passively address errors in the
𝒚
\bm{y}
-space by enhancing the robustness of the neural network. For example,
Krainovic et al. 2023
suggests that introducing jittering in the
𝒚
\bm{y}
-space can achieve robustness in worst-case
ℓ
2
\ell_{2}
perturbations. In
Hu et al. 2023
, a deep equilibrium solver is trained using various incorrect
𝒜
\mathcal{A}
’s, demonstrating greater robustness to variations in
𝒜
\mathcal{A}
compared to the Plug-and-Play method. However, it is noteworthy that such robust networks often experience a tradeoff between accuracy and stability, as discussed in
Krainovic et al. 2023
;
Gottschling et al. 2023
.
In contrast, our proposed approach involves an active strategy for mitigating model mismatch. The experimental results in Section
6
illustrate the effectiveness of our method compared to the same LU network that merely learns a robust mapping passively.
Various active forward model matching algorithms are tailored to different tasks. For linear IPs, model mismatches can be more easily resolved by alternatively updating the model parameters and the underlying signal. For instance,
Fergus et al. 2006
;
Cai et al. 2009
;
Levin et al. 2009
;
Cho & Lee 2009
reconstructed the latent images with updates in the forward model solved directly from least squares. However, these methods are often constrained to linear scenarios and rely on task-specific recovery techniques, which are typically highly sensitive to hyperparameters
Cho & Lee 2009
. Later works
Nan & Ji 2020
;
Gilton et al. 2021b
learn a regularizer from data, resulting in improved performance on linear inverse problems.
The structure of the initial forward model in a nonlinear inverse problem often differs from the true model. For instance, linear models may be used to estimate nonlinear functions. This motivates the use of neural networks to learn the true forward model
(
Candiani et al. 2021
;
Koponen et al. 2021
;
Lunz et al. 2021
)
. However, traditional neural network training approaches require separate stages for learning the forward model and signal reconstruction. This requires a substantial amount of training data to ensure accurate mapping
(
Arjas et al. 2023
)
from any
𝒙
{\bm{x}}
(which represents ground truth and intermediate reconstructions) to its corresponding
𝒚
{\bm{y}}
in iterative reconstructions. With limited training data, this method falls out of the discussion.
In contrast, we introduce an untrained residual network capable of handling forward model mismatch for general IPs. The proposed method is a unified framework that applies to both linear and nonlinear IPs, offering enhanced stability during training, requiring no additional data, and allowing fitting the forward model and reconstruction simultaneously in a single pass. We further list our contributions as follows:
•
General model-based architecture in handling model-mismatch:
We present a general model-based architecture for solving IPs when only an approximation of the forward model is available. Unlike classical model-based architectures which require precise knowledge of
𝒜
\mathcal{A}
, we propose two variants of the model-based algorithms that can iteratively update the forward model along with the reconstruction.
•
Introducing untrained neural network for model-mismatch:
We introduce a novel approach by integrating an untrained neural network into a model-based architecture to handle model-mismatch in both linear and nonlinear inverse problems. By updating its parameters alongside the reconstruction process, we circumvent the need for additional data and pre-training the forward model in solving nonlinear inverse problems. Indeed, we show that random initialization of this network during evaluation yields the same level of reconstruction.
•
Proof of convergence and empirical validation:
We establish the convergence of the proposed algorithm under mild conditions and empirically verify its convergence using a Deep Equilibrium Model (DEQ) structure.
•
Improved performance in linear and nonlinear IP tasks:
In contrast to the passive robust training approaches within general IP solvers using model-based architectures, our methods demonstrate substantial improvement in image blind deblurring (linear), seismic blind deconvolution (linear), and landscape defogging (nonlinear) tasks. It’s worth noting that while our application showcases improvements in these specific areas, its scope extends beyond to fields like medical imaging.
•
Effectiveness in iterative reconstructions and robustness:
We show that proposed algorithms lead to more effective intermediate reconstructions. They also exhibit robustness to different random initializations of the residual block and are more stable in scenarios involving a higher number of iterations during evaluation.

## Method

The discussion above has assumed exact knowledge of
𝒜
\mathcal{A}
. Here we now suppose that we have an initial guess of the forward model
𝒜
0
\mathcal{A}_{0}
and consider a neural network
f
θ
f_{\theta}
that learns the measurement residual due to model misfit given the signal
𝒙
\bm{x}
and
𝒜
0
\mathcal{A}_{0}
, i.e.,
𝒚
=
𝒜
⁡
(
𝒙
)
+
ϵ
=
𝒜
0
​
(
𝒙
)
+
f
θ
​
(
𝒙
,
𝒜
0
)
+
ϵ
.
\bm{y}=\mathcal{A}(\bm{x})+\bm{\epsilon}=\mathcal{A}_{0}(\bm{x})+f_{\theta}(\bm{x},\mathcal{A}_{0})+\bm{\epsilon}.
We assume that
𝒜
0
\mathcal{A}_{0}
is a useful estimate of the true forward model, in other words,
𝒜
⁡
(
𝒙
)
\mathcal{A}(\bm{x})
and
𝒜
0
​
(
𝒙
)
\mathcal{A}_{0}(\bm{x})
are close. We can then express the optimization problem in (
2
) as
min
𝒙
,
θ
⁡
1
2
​
‖
𝒚
−
𝒜
0
​
(
𝒙
)
−
f
θ
​
(
𝒙
,
𝒜
0
)
‖
2
2
+
γ
​
r
​
(
𝒙
)
+
τ
​
‖
f
θ
​
(
𝒙
,
𝒜
0
)
‖
2
2
.
\begin{split}\mathop{\text{min}}_{\bm{x},\theta}~\frac{1}{2}\|\bm{y}-\mathcal{A}_{0}(\bm{x})-f_{\theta}(\bm{x},\mathcal{A}_{0})\|^{2}_{2}+\gamma r(\bm{x})+\tau\|f_{\theta}(\bm{x},\mathcal{A}_{0})\|^{2}_{2}.\end{split}
(5)
Introducing an auxiliary variable
𝒛
\bm{z}
, the solution of (
5
) is equivalent to solving
min
𝒙
,
𝒛
,
θ
⁡
1
2
​
‖
𝒚
−
𝒜
0
​
(
𝒛
)
−
f
θ
​
(
𝒛
,
𝒜
0
)
‖
2
2
+
γ
​
r
​
(
𝒙
)
+
τ
​
‖
f
θ
​
(
𝒛
,
𝒜
0
)
‖
2
2
+
λ
​
‖
𝒙
−
𝒛
‖
2
2
.
\begin{split}\mathop{\text{min}}_{\bm{x},\bm{z},\theta}~\frac{1}{2}\|\bm{y}-\mathcal{A}_{0}(\bm{z})-f_{\theta}(\bm{z},\mathcal{A}_{0})\|^{2}_{2}+\gamma r(\bm{x})+\tau\|f_{\theta}(\bm{z},\mathcal{A}_{0})\|^{2}_{2}+\lambda\|\bm{x}-\bm{z}\|^{2}_{2}.\end{split}
(6)
Similar to the HQS updates, we first initialize
𝒙
0
\bm{x}_{0}
,
𝒛
0
\bm{z}_{0}
, and
θ
0
\theta_{0}
. We then update each variable in the objective function by keeping other variables fixed. For
k
=
1
,
2
,
…
,
K
k=1,2,\ldots,K
, we have
𝒛
k
+
1
=
arg
​
min
z
⁡
1
2
​
‖
𝐲
−
𝒜
0
​
(
𝐳
)
−
f
θ
k
​
(
𝐳
,
𝒜
0
)
‖
2
2
+
τ
​
‖
f
θ
k
​
(
𝐳
,
𝒜
0
)
‖
2
2
+
λ
​
‖
𝐱
k
−
𝐳
‖
2
2
,
θ
k
+
1
=
arg
​
min
θ
⁡
1
2
​
‖
𝐲
−
𝒜
0
​
(
𝐳
k
+
1
)
−
f
θ
​
(
𝐳
k
+
1
,
𝒜
0
)
‖
2
2
+
τ
​
‖
f
θ
​
(
𝐳
k
+
1
,
𝒜
0
)
‖
2
2
,
𝒙
k
+
1
=
prox
γ
2
​
λ
,
r
​
(
𝒙
k
−
η
⁡
(
𝒙
k
−
𝒛
k
+
1
)
)
.
\begin{split}\bm{z}_{k+1}&=\argmin_{z}~\frac{1}{2}\|\bm{y}-\mathcal{A}_{0}(\bm{z})-f_{\theta_{k}}(\bm{z},\mathcal{A}_{0})\|^{2}_{2}+\tau\|f_{\theta_{k}}(\bm{z},\mathcal{A}_{0})\|^{2}_{2}+\lambda\|\bm{x}_{k}-\bm{z}\|^{2}_{2},\\
\theta_{k+1}&=\argmin_{\theta}~\frac{1}{2}\|\bm{y}-\mathcal{A}_{0}(\bm{z}_{k+1})-f_{\theta}(\bm{z}_{k+1},\mathcal{A}_{0})\|^{2}_{2}+\tau\|f_{\theta}(\bm{z}_{k+1},\mathcal{A}_{0})\|^{2}_{2},\\
\bm{x}_{k+1}&=\text{prox}_{\frac{\gamma}{2\lambda},r}(\bm{x}_{k}-\eta(\bm{x}_{k}-\bm{z}_{k+1})).\end{split}
(7)
It should be noted that the update on
𝒛
\bm{z}
no longer has a closed-form solution due to the nonlinearity of
f
θ
f_{\theta}
. However, this can be efficiently computed using Autograd in Pytorch
(
Paszke et al. 2017
)
or other differentiation computing algorithms. Meanwhile, the update on
θ
\theta
follows the regular backpropagation for network parameters. We can then replace the proximal operator with a neural network, and the update on
𝒙
\bm{x}
connects all the components to form an LU network as illustrated in Fig.
2
. The parameters in the proximal network, are learned through end-to-end training, where
η
\eta
is a trainable step size.
While we refer to
f
θ
f_{\theta}
as a network in the context of learning forward model mismatch, it differs from classical training approaches where weights remain fixed during evaluation. In our framework, the parameters
θ
\theta
in
f
f
are updated both during training and evaluation for each instance. The objective is to iteratively align the data fidelity term for a specific measurement
𝒚
\bm{y}
through a nonlinear estimation
f
f
with the minimal
ℓ
2
\ell_{2}
-norm. Our proposed technique involves using untrained (random) weights
θ
0
\theta_{0}
at the initial iteration and adjusting
θ
k
\theta_{k}
based on the loss outlined in the second line of (
7
). The concept of employing an untrained neural network is reminiscent of Deep Image Prior
(
Ulyanov et al. 2018
)
, which initializes a reconstruction network randomly, treating the weights as an implicit prior for the reconstruction. In our approach, however, we suggest using an untrained neural network as an integral part of the reconstruction process to address model mismatch while still being capable of learning regularization updates from all training data. Subsequent experiments demonstrate that during evaluation, the residual network
f
f
can be initialized with various untrained weights, resulting in different performance levels.
Note that the proposed iterative updates look reminiscent of the alternating direction method of multipliers (ADMM), like
Yang et al. 2016
. When the forward model is precisely known, ADMM aims to minimize the Lagrangian of the objective function with an additional auxiliary variable. The proposed algorithm uses HQS as a simpler method to split variables in solving an optimization problem, but the algorithm could easily be generalized to other solvers such as ADMM.
Figure 2
:
Illustration of the
k
t
​
h
k^{th}
iteration of an
𝒜
\mathcal{A}
-adaptive LU network.
𝒙
0
\bm{x}_{0}
is fed into the network, the auxiliary update and the correction update corresponding to updates in
θ
\theta
and
𝒛
\bm{z}
respectively, and the proximal network in green is updated using end-to-end training. The final output contains the parameters for estimating the function mismatch and the reconstruction estimate.
