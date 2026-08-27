# Double Blind Imaging with Generative Modeling

paper_id: arxiv:2503.21501v1
tier: H
source_used: html_arxiv
warning: none

## Intro

Computational imaging techniques are integral across a wide range of disciplines, including astronomy
et The EHT Collaboration al. 2019
;
Vojtekova et al. 2020
, microscopy
Rivenson et al. 2017
;
Chowdhury et al. 2019
, medicine
Reader et al. 2021
;
Heckel et al. 2024
, and consumer electronics
Delbracio et al. 2021
. Under additive noise, the imaging system can be modeled as
y
=
A
⁡
(
x
)
+
η
,
y=A(x)+\eta,
(1)
where
x
∈
ℂ
n
x\in\mathbb{C}^{n}
is the image,
A
:
ℂ
n
→
ℂ
m
A:\mathbb{C}^{n}\rightarrow\mathbb{C}^{m}
represents the measurement (forward) operator,
η
∈
ℂ
m
\eta\in\mathbb{C}^{m}
is random additive noise, and
y
∈
ℂ
m
y\in\mathbb{C}^{m}
are the measurements.
The central challenge is to estimate the clean image
x
x
from measurements
y
y
, a task known as the inverse problem. This problem is typically ill-posed due to additive noise (
η
\eta
) and potentially insufficient measurements (
m
<
n
m<n
) for stable inversion.
Numerous classical
Donoho 2006
;
Lustig et al. 2007
and deep learning-based methods
Gregor and LeCun 2010
;
Bora et al. 2017
;
Kawar et al. 2022
;
Ongie et al. 2020
have been proposed to address signal recovery. These methods commonly assume that the measurement process
A
A
is known a priori, allowing it to be used directly in the recovery algorithm. This assumption implies that the exact characteristics of the imaging system used to acquire the measurements are known.
Figure 1
:
Example of imaging system setup where the measurement process is unknown due to hardware, environment, or scene variables.
D
ϕ
D_{\phi}
z
z
G
θ
G_{\theta}
Image dataset
{
x
1
,
x
2
,
⋯
,
x
N
}
\{x_{1},\ x_{2},\ \cdots,x_{N}\}
Measurement dataset
{
y
1
,
y
2
,
⋯
,
y
N
}
\{y_{1},\ y_{2},\ \cdots,y_{N}\}
κ
g
\kappa^{g}
x
r
x^{r}
𝒜
\mathcal{A}
y
r
y^{r}
y
g
y^{g}
Figure 2
:
Schematic block diagram illustrating our algorithm for unsupervised learning of the imaging system parameters. We learn a generative network
G
θ
G_{\theta}
that generates system parameters
κ
g
\kappa^{g}
. Given a set of training images
{
x
1
,
x
2
,
⋯
,
x
N
}
\{x_{1},x_{2},\cdots,x_{N}\}
, we pass these images along with the parameters
κ
g
\kappa^{g}
through the imaging system to generate measurements
y
g
y^{g}
. The discriminative network
D
ϕ
D_{\phi}
is trained to distinguish between the generated measurements
y
g
y^{g}
and actual measurements
y
r
∼
{
y
1
,
y
2
,
⋯
,
y
N
}
y^{r}\sim\{y_{1},y_{2},\cdots,y_{N}\}
. Note that the image dataset and measurement dataset are
unpaired
, independent, and disjoint of one another.
Blind Inverse Problems.
In many practical scenarios, the imaging system is not known exactly at the time of signal recovery, but can be modeled as belonging to a family of forward models
A
κ
A_{\kappa}
, where
κ
\kappa
parameterizes the unknown components.
The uncertainty in the imaging system can stem from various sources, and can often be modeled statistically. Randomness may be introduced by the imaging hardware itself, such as fixed but unknown lens aberrations. Uncertainty may also arise from hardware-agnostic effects like camera shake
Delbracio and Sapiro 2015
, motion blur due to scene dynamics
Gong et al. 2017
, or atmospheric turbulence
Mao et al. 2020
. In these cases, even if the imaging hardware is static, the overall system is not, necessitating estimation of the imaging system for each measurement capture.
Model-based algorithms for blind inverse problems jointly estimate both the clean image
x
x
and the correct forward model
A
κ
A_{\kappa}
corresponding to the measurement data
y
y
Li et al. 2017
;
Ling and Strohmer 2018
;
Krishnan et al. 2011
;
Wipf and Zhang 2013
;
Ahmed et al. 2014
. This is typically accomplished by imposing priors on the image,
p
⁡
(
x
)
p(x)
, and the measurement model,
p
⁡
(
A
)
p(A)
Chung et al. 2023
.
While it is often feasible to learn
p
⁡
(
x
)
p(x)
through generative modeling using available clean images
x
∼
p
⁡
(
x
)
x\sim p(x)
, directly observing or learning
p
⁡
(
A
)
p(A)
is more challenging. This is because the imaging system itself is not directly observed; rather, only the measurements
y
∼
p
⁡
(
y
)
y\sim p(y)
that follow Eq.
1
. Moreover, it is not practical to assume a training set of paired data of images and measurements, i.e.
{
(
x
i
,
y
i
)
}
i
∈
[
N
]
\{(x_{i},y_{i})\}_{i\in[N]}
, which further complicates the learning of the imaging system.
Our approach.
We are interested in the setting where we do not have access to direct observations of the forward process or paired data. We assume there is some low-dimensional structure in the forward process, and we assume access to an independent training set of clean images, and an independent training set of measurements.
We build upon the principles of AmbientGAN
Bora et al. 2018
, but with a crucial distinction: instead of learning the distribution of clean images, we learn the distribution of forward operators using the unpaired images and measurements as training data. Our underlying assumption is that the measurement process can be represented implicitly by a generative model due to some underlying low-dimensional structure.
This approach to learning the imaging system from unpaired examples naturally integrates with downstream system ID tasks that have model uncertainty (e.g. in blind deconvolution). Traditional blind deconvolution methods rely on alternating or joint estimation of the image and the blur kernel by assuming both are unknown and iteratively refining both components
Krishnan et al. 2011
;
Wipf and Zhang 2013
;
Ahmed et al. 2014
. With our generative model-based framework, the learned distribution of forward operators
p
⁡
(
A
)
p(A)
can be used to provide a robust prior for the measurement operator in these solvers. This prior can guide off-the-shelf blind deconvolution algorithms by constraining the possible forms of the blur kernel, improving convergence to accurate solutions even when no exact operator measurements are available
Chung et al. 2023
. Our approach acts as a flexible plug-in for downstream tasks that require a statistical model for the imaging system.
Our Contributions
1.
We propose an unsupervised algorithm that learns a prior distribution over forward operators using only
unpaired
clean images and corrupted measurements that are independent and disjoint from one-another. Forward operators are structured operators defined by a physical imaging system, and in this paper we consider Gaussian smoothing and motion blur, though other structure can also be incorporated. We design a network architecture for the generative model which respects the physics of the imaging system and then leverage the loss function in AmbientGAN
Bora et al. 2018
to train the model.
2.
We experimentally show that our method is robust to additive noise in the imaging system.
The experiments in Section
4.3
show that our method is robust in low-SNR settings.
3.
Our learning algorithm is independent of the downstream system ID or signal recovery algorithm. In Section
4.4
, we show that this modular approach of learning system priors can be used in off-the-shelf state-of-the-art algorithms
Chung et al. 2023
for blind deconvolution.
4.
In Section
4.4
, we evaluate our algorithm on the classical problems of Gaussian and motion deblurring. We find that training with imaging forward processes learned with our technique enables reconstruction quality close to that of models trained with direct access to imaging system forward process examples.
These contributions provide a versatile and robust tool for enabling blind inverse imaging problem solvers, addressing limitations of previous models that rely on direct observations of measurement systems.
While we focus on the application of blind deblurring here, our framework has the potential to be applied to more general structured blind inverse problems.

## Method

As stated above, many SOTA approaches to solving blind inverse problems require information about the prior over possible measurement operators
p
⁡
(
A
)
p(A)
Chung et al. 2023
;
Laroche et al. 2023
. We propose learning this from only unpaired samples of clean images and corrupted measurements. In other words, we use samples from the marginal distributions
x
i
∼
p
⁡
(
x
)
x_{i}\sim p(x)
and
y
j
∼
p
⁡
(
y
)
y_{j}\sim p(y)
respectively where it is assumed that
y
=
A
κ
​
(
x
)
+
η
y=A_{\kappa}(x)+\eta
(7)
where
A
κ
∼
p
⁡
(
A
κ
)
A_{\kappa}\sim p(A_{\kappa})
, and
η
\eta
is additive noise of a known family (Gaussian with known variance in this work). It is also assumed that
A
κ
A_{\kappa}
and
x
x
are independent random variables.
A schematic diagram of our approach is shown in Figure
2
. Our method approximates
p
⁡
(
A
κ
)
p(A_{\kappa})
implicitly by learning how to match the measurement distribution
p
⁡
(
y
)
p(y)
by selecting an
x
i
x_{i}
from the clean dataset and passing it through a synthetic measurement system sampled from some generator function
A
κ
=
G
θ
​
(
z
)
A_{\kappa}=G_{\theta}(z)
, where
z
∼
𝒩
⁡
(
0
,
I
)
z\sim\mathcal{N}(0,I)
. Importantly, we assume some structure about
A
A
is known, e.g. through the parameterization
κ
\kappa
. For example, for blind deconvolution,
κ
\kappa
defines a
k
k
-dimensional convolution kernel.
More specifically, we use the following GAN inspired objective:
min
G
θ
⁡
max
D
ϕ
​
𝔼
y
∼
p
⁡
(
y
)
​
[
ln
⁡
(
D
ϕ
​
(
y
)
)
]
\displaystyle\min_{G_{\theta}}\max_{D_{\phi}}\mathbb{E}_{y\sim p(y)}\left[\ln(D_{\phi}(y))\right]
+
𝔼
z
∼
p
z
​
(
z
)
,
x
∼
p
⁡
(
x
)
,
η
∼
p
⁡
(
η
)
​
[
ln
⁡
(
1
−
D
ϕ
​
(
G
θ
​
(
z
)
​
x
+
η
)
)
]
.
\displaystyle+\mathbb{E}_{z\sim p_{z}(z),x\sim p(x),\eta\sim p(\eta)}\left[\ln(1-D_{\phi}(G_{\theta}(z)x+\eta))\right].
(8)
Note here that the discriminator does not incentive the output of the generative process to create realistic images but rather it pushes to learn realistic measurements which may be corrupted images (e.g., blurred, noised, etc.). The objective above is similar to that of AmbientGAN
Bora et al. 2018
, except that we learn the generator of imaging systems (degradations processes) instead of clean distributions of signals.
Due to the implicit nature of GAN training, a natural solution to learning in the presence of additive noise can be achieved. This is done by simply adding the correct amount of random noise at training time to the generated measurements from the synthetic imaging system. This implicitly deconvolves the noisy measurement distribution we learn to match with the additive noise distribution used to corrupt our real measurement data distribution.
Figure 3
:
BDPS reconstructions for blind Gaussian (top) and motion (bottom) deblurring on AFHQ
128
×
128
128\times 128
. Using various kernel priors.
