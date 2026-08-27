# Integrating Bayesian Spectral Deconvolution and Expert Scientific Reasoning for Robust Peak Estimation

paper_id: arxiv:2605.17518v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Spectral deconvolution estimates the number, position, and variance of peaks by regressing the spectra obtained by irradiating materials with X-rays or visible light as a sum of basis functions. This method is crucial for evaluating material properties and chemical structures.
Mass spectrometry
12
is used to determine the molecular weight, whereas infrared (IR) spectroscopy
8
;
4
and Raman spectroscopy
19
are used to investigate the physical properties and chemical structure of synthesized compounds. Spectral deconvolution enables the identification of the molecular weight and partial structures.
Molecular bonding can be evaluated using the nuclear magnetic resonance spectra
5
, whereas X-ray diffraction (XRD) determines the lattice constants and atomic arrangements in crystals
14
by obtaining and deconvolving the spectra.
The properties and chemical structures of materials are determined by measuring the spectra and deconvoluting them into their components.
The automation of processes such as material synthesis and spectral measurement has led to an increase in data requiring spectral deconvolution
3
. Despite these advancements, spectral analysis continues to depend to some extent on expert scientific interpretation
1
;
28
. The significance of spectral deconvolution, which seeks to promote objectivity and automation within an information science framework, is therefore being increasingly acknowledged.
Existing spectral deconvolution methods may struggle to determine spectral models, such as the number of peaks or the basis function model, for spectral data with complex peak shapes and overlapping peaks.
Estimating the number and shape of peaks is a fundamental requirement for achieving spectral deconvolution.
To improve this capability, Bayesian spectral deconvolution, which applies Bayesian inference to the problem, has been proposed.
Bayesian spectral deconvolution constructs a statistical model by applying Bayesian inference to a regression model that represents the spectrum as a sum of basis functions. This approach enables the estimation of spectral deconvolution model parameters, including the optimal number of peaks required to represent the spectral data
17
.
Undesirable components such as high-intensity noise or substance-derived background often appear in spectral data owing to instrument errors, signal processing, or the measured sample and are problematic for analysis
9
;
31
.
In fields lacking a well-defined physical model, the background model may not be well defined, complicating the spectral deconvolution analysis in some cases
6
.
Model selection for various analyses, such as peak-number estimation, becomes difficult when the spectral data contain high-intensity noise or background
21
in conventional Bayesian spectral deconvolution.
Thus, existing methods have the drawback that spectral deconvolution becomes difficult when noise dominates or when the background component model is unknown.
However, researchers may achieve spectral deconvolution by applying domain-specific knowledge, even in scenarios where noise is prevalent or the background component model is not defined.
Scientists do not rely solely on the spectral data to analyze the spectra. Rather, they analyze the spectra in the context of the physical properties and chemical structure of the material and variations observed in other measurements collected during their analysis.
Correlations between the changes in material properties or chemical structure often result in shifts in the spectral peaks. These shifts help in identifying the key candidate peaks for analysis and highlight the spectral regions warranting further investigation.
Scientists have achieved spectral deconvolution by leveraging the insights gained from such knowledge, enabling them to extract peak structures buried in noise or appearing against unknown backgrounds
10
.
Consequently, peak-number estimation and other spectral model selections are achievable even for complex spectral shapes
32
.
The analytical approach used by scientists to focus on the important spectral regions based on the physical property values is crucial for analyzing spectra dominated by noise or where background components are not modeled.
We propose a spectral deconvolution framework that models the process followed by scientists, namely, using material properties and structures to identify the key regions, enabling robust peak estimation even in the presence of dominant noise or unknown backgrounds.
Several methods based on such scientific practices have already been developed.
A Bayesian analysis method for X-ray absorption near-edge structure spectra has been proposed, which improves the accuracy by assigning different model structures or prior distributions to each region, based on the knowledge that the peak-shape measurements differ across the energy regions
16
.
Other studies have applied regression modeling to the relationship between the physical property values and peak structures derived from spectral deconvolution
11
.
However, these approaches merely provide a framework in which spectral data analysis and knowledge extraction from auxiliary measurement data, such as physical properties, are conducted separately.
Integrating these two steps is expected to automate spectral deconvolution while enabling more comprehensive knowledge extraction of the relationship between the physical properties and spectra.
In this context, our proposed framework comprehensively models the analytical process followed by scientists for deconvoluting complex spectra, as follows.
First, we employ the Bayesian spectral deconvolution model mentioned above to model the spectral data.
Next, we model the practice of examining the relationship between peak structures derived from deconvolution and the physical properties or chemical structures as Gaussian process regression.
This approach aims to integrate the previously underutilized physical property information into the spectral deconvolution, enabling accurate model selection and parameter estimation even in complex spectra containing noise and background.
The proposed method was applied to the spectral deconvolution of synthetic spectral data and IR spectra of polylactic acid, a representative biodegradable polymer, to verify its effectiveness.

## Method

II.1
Problem formulation
We address the problem of selecting spectral models, such as the number of peaks, by integrating the spectral data with physical prior knowledge. This knowledge includes the physical property value, as used by scientists.
We define the dataset used in the proposed method.
We consider the spectral data
{
x
i
,
y
i
}
i
=
1
N
\{x_{i},y_{i}\}_{i=1}^{N}
consisting of
N
N
observation points.
x
i
x_{i}
represents the horizontal axis, such as wavenumber or wavelength;
y
i
y_{i}
represents the spectral intensity at that point.
We define a spectral dataset as
Y
:=
{
y
i
}
i
=
1
N
Y:=\{y_{i}\}_{i=1}^{N}
. We assume
N
′
N^{\prime}
spectral datasets
Y
j
:=
{
y
i
,
j
}
i
=
1
N
Y_{j}:=\{y_{i,{j}}\}_{i=1}^{N}
(
j
=
1
,
…
,
N
′
j=1,\dots,N^{\prime}
) sharing the observation points
{
x
i
}
i
=
1
N
\{x_{i}\}_{i=1}^{N}
but representing distinct physical states, such as datasets containing material properties.
We define the paired dataset as
𝒟
=
{
(
Y
j
obs
,
z
j
)
}
j
=
1
N
′
\mathcal{D}=\{(Y_{j}^{\mathrm{obs}},z_{j})\}_{j=1}^{N^{\prime}}
, where
Y
j
obs
Y_{j}^{\mathrm{obs}}
is the observed spectrum of sample
j
j
and
z
j
z_{j}
is the corresponding physical property.
This dataset
𝒟
\mathcal{D}
constitutes the dataset assumed by the proposed method.
We formulate the problem of selecting a spectral model
M
M
by combining the spectral data
𝒴
\mathcal{Y}
and physical prior knowledge
Z
Z
(e.g., material properties), as follows.
arg
​
max
M
⁡
p
⁡
(
M
|
Z
,
𝒴
)
\displaystyle\mathop{\rm arg\penalty\ max}\limits_{M}p({M}|Z,\mathcal{Y})
(1)
Note that we first formulate the problem in terms of the full Bayesian model posterior.
As shown below, this posterior can be decomposed into a spectrum-only evidence term and a physical-property-consistency term.
The proposed method then adopts the latter term as the model-selection criterion, in accordance with the aim of this study.
II.2
Model
This section presents a Bayesian hierarchical model for spectral deconvolution that incorporates physical property values, mirroring scientific practices (Figure
1
).
The proposed framework consists of two layers. The first is a spectral deconvolution model analyzing the spectrum. The second is a regression model that relates the physical property values to the spectral data.
After describing the individual layers, we explain the spectral deconvolution model of the process followed by scientists.
II.2.1
Spectral deconvolution model
The spectral deconvolution model represents the spectral data as a linear combination of basis functions, such as Gaussian functions.
The regression of spectral data using the spectral deconvolution model yields information such as the number of peaks, peak positions, and peak variances
17
.
Consider the spectral data consisting of
N
N
observations
{
x
i
,
y
i
}
i
=
1
N
\{x_{i},y_{i}\}_{i=1}^{N}
.
The spectral deconvolution model, utilizing Gaussian basis functions subject to independent and identically distributed (i.i.d.) normal noise, is expressed as follows:
Figure 1
:
Graphical model of the proposed method.
M
M
denotes the spectral model. For each sample
j
j
, the spectral parameter
θ
j
\theta_{j}
defines both the posterior predictive spectrum
Y
j
Y_{j}
and the observed spectrum
Y
j
obs
Y_{j}^{\mathrm{obs}}
through the spectral likelihood model. The physical property
Z
Z
is modeled by Gaussian process regression using the latent function value
F
F
conditioned on the spectral information. The red (blue) box indicates the components of spectral deconvolution (Gaussian process regression).
p
⁡
(
Y
|
θ
)
\displaystyle p(Y|\theta)
=
\displaystyle=
∏
i
=
1
N
1
2
​
π
​
σ
​
exp
⁡
(
−
(
y
i
−
g
⁡
(
x
i
,
θ
)
)
2
2
​
σ
2
)
.
\displaystyle\prod_{i=1}^{N}\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y_{i}-g(x_{i};\theta))^{2}}{2\sigma^{2}}\right).
(2)
g
g
denotes the spectral fitting function,
θ
\theta
the regression parameters, and
σ
\sigma
the standard deviation of the noise. The representative examples of
g
g
include
g
⁡
(
x
,
θ
)
\displaystyle g(x;\theta)
=
\displaystyle=
∑
j
=
1
M
w
j
​
exp
⁡
(
−
(
x
−
μ
j
)
2
2
​
a
j
2
)
.
\displaystyle\sum_{j=1}^{M}w_{j}\exp\left(-\frac{(x-\mu_{j})^{2}}{2a_{j}^{2}}\right).
(3)
M
M
denotes the number of peaks in the fitting model, and
θ
=
{
w
j
,
μ
j
,
a
j
}
j
=
1
M
\theta=\{w_{j},\mu_{j},a_{j}\}_{j=1}^{M}
represents the set of parameters.
II.2.2
Physical-property regression model
The physical-property regression model connects the spectral data to the corresponding physical property values.
We employed Gaussian process regression as the Bayesian physical-property regression model in this study.
In the physical-property regression model, a single sample constitutes a pair of spectral data
Y
:=
{
y
i
}
i
=
1
N
Y:=\{y_{i}\}_{i=1}^{N}
and a physical property value
z
z
.
This can be extended to multiple samples, as follows. Given a set of spectral and measurement data of sample size
N
′
N^{\prime}
,
{
𝒴
,
Z
}
:=
{
Y
j
,
z
j
}
j
=
1
N
′
\{\mathcal{Y},Z\}:=\{Y_{j},z_{j}\}_{j=1}^{N^{\prime}}
, the Gaussian process regression model
2
is formulated as
p
⁡
(
Z
|
F
,
𝒴
)
=
(
α
2
​
π
)
N
′
2
​
exp
⁡
[
−
1
2
​
(
Z
−
F
)
T
​
α
​
I
​
(
Z
−
F
)
]
.
\displaystyle{\it p}(Z|F,\mathcal{Y})=\left({\frac{\alpha}{2\pi}}\right)^{\frac{N^{\prime}}{2}}\exp\left[-\frac{1}{2}\left(Z-F\right)^{T}\alpha I\left(Z-F\right)\right].
(4)
We define
F
:=
F
⁡
(
𝒴
)
=
(
f
⁡
(
Y
1
)
,
…
,
f
⁡
(
Y
N
′
)
)
F:=F(\mathcal{Y})=(f(Y_{1}),\dots,f(Y_{N^{\prime}}))
. The prior distribution
p
⁡
(
F
)
p(F)
is assumed to be an
N
′
N^{\prime}
-dimensional normal distribution with zero mean and covariance matrix
K
K
. The hyperparameter
α
\alpha
denotes the noise precision, which is the inverse of the noise variance in the likelihood model.
p
⁡
(
F
)
=
(
1
(
2
​
π
)
N
′
​
|
K
|
)
1
/
2
​
exp
⁡
(
−
1
2
​
F
T
​
K
−
1
​
F
)
\displaystyle p(F)=\left(\frac{1}{(2\pi)^{N^{\prime}}|K|}\right)^{1/2}\exp\left(-\frac{1}{2}F^{T}K^{-1}F\right)
(5)
Here,
k
⁡
(
Y
i
,
Y
j
)
k(Y_{i},Y_{j})
denotes a kernel function representing the similarity between
Y
i
Y_{i}
and
Y
j
Y_{j}
.
II.2.3
Model of spectral deconvolution process followed by scientists
We modeled the method followed by scientists for extracting high-precision information by analyzing the spectral data in combination with physical prior knowledge.
The simplest physical prior knowledge is the value of a property related to a spectrum.
Given the spectral data and corresponding physical properties, scientists analyze the spectral features, such as specific regions or peaks, that are correlated with variations in the physical properties.
The decision of scientists to focus on specific regions while down-weighting others corresponds to the operation of extracting only the dimensions of relevant variables.
Thus, the above represents a variable selection process in regression analysis, where the spectral intensities at each wavenumber serve as explanatory variables predicting the physical property values.
To reflect this requirement in the model, it is effective to model a predictive relationship in which spectral structures serve as explanatory variables for the physical property values.
Incorporating this perspective into the model, we treat spectral features as predictors of the physical properties, so that spectral components relevant to the target property are preferentially weighted during model selection. Introducing a causal link from the spectra to the physical properties enables predictive performance to guide spectral-structure estimation, ensuring that physically meaningful components are prioritized.
By mirroring this scientific spectral deconvolution process, the causal relationships can be modeled as depicted in the graphical model in Figure
1
.
Conventional spectral deconvolution methods focus solely on estimating the number of peaks and their parameters from the spectral data, without accounting for their causal relationships with the physical properties.
In contrast, the spectral deconvolution method followed by scientists estimates the number of peaks and their parameters while maintaining a causal link between the spectral data and physical property values.
This represents a causal relationship in which specifying a spectral model (e.g., the number of peaks) and parameters (e.g., positions and variances) generates a spectrum from which the measured value
z
z
is derived.
The following statistical model is derived from this graphical model.
p
⁡
(
M
,
ϑ
,
𝒴
,
𝒴
obs
,
F
,
Z
)
\displaystyle p(M,\vartheta,\mathcal{Y},\mathcal{Y}^{\text{obs}},F,Z)
=
\displaystyle=
p
⁡
(
Z
|
F
,
𝒴
)
​
p
​
(
F
)
​
p
​
(
𝒴
|
ϑ
)
\displaystyle{\it p}(Z|F,\mathcal{Y}){\it p}(F){\it p}(\mathcal{Y}|\vartheta)
(6)
×
p
⁡
(
𝒴
obs
|
ϑ
)
​
p
​
(
ϑ
|
M
)
​
p
​
(
M
)
.
\displaystyle\times{\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta){\it p}(\vartheta|M){\it p}(M).
Here, let
ϑ
:=
{
θ
j
}
j
=
1
N
′
\vartheta:=\{\theta_{j}\}_{j=1}^{N^{\prime}}
; then,
p
⁡
(
𝒴
|
ϑ
)
,
p
⁡
(
𝒴
obs
|
ϑ
)
,
p
⁡
(
ϑ
|
M
)
{\it p}(\mathcal{Y}|\vartheta),{\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta),{\it p}(\vartheta|M)
are defined as follows:
p
⁡
(
𝒴
|
ϑ
)
:=
∏
j
=
1
N
′
p
⁡
(
Y
j
|
θ
j
)
,
\displaystyle{\it p}(\mathcal{Y}|\vartheta):=\prod_{j=1}^{N^{\prime}}{\it p}(Y_{j}|\theta_{j}),
(7)
p
⁡
(
𝒴
obs
|
ϑ
)
:=
∏
j
=
1
N
′
p
⁡
(
Y
j
obs
|
θ
j
)
,
\displaystyle{\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta):=\prod_{j=1}^{N^{\prime}}{\it p}(Y^{\mathrm{obs}}_{j}|\theta_{j}),
(8)
p
⁡
(
ϑ
|
M
)
:=
∏
j
=
1
N
′
p
⁡
(
θ
j
|
M
)
.
\displaystyle{\it p}(\vartheta|M):=\prod_{j=1}^{N^{\prime}}{\it p}(\theta_{j}|M).
(9)
Given the spectral data
𝒴
obs
\mathcal{Y}^{\mathrm{obs}}
and property values
Z
Z
, the posterior
p
⁡
(
M
|
Z
,
𝒴
obs
)
p(M|Z,\mathcal{Y}^{\mathrm{obs}})
for the spectral model
M
M
is obtained by marginalizing the joint probability.
p
⁡
(
M
|
Z
,
𝒴
obs
)
\displaystyle p(M|Z,\mathcal{Y}^{\mathrm{obs}})
∝
p
⁡
(
Z
,
𝒴
obs
,
M
)
\displaystyle\propto p(Z,\mathcal{Y}^{\mathrm{obs}},M)
=
∫
d
​
F
​
𝑑
𝒴
​
𝑑
ϑ
​
p
​
(
Z
,
F
,
𝒴
,
𝒴
obs
,
ϑ
,
M
)
\displaystyle=\int dFd\mathcal{Y}d\vartheta p(Z,F,\mathcal{Y},\mathcal{Y}^{\mathrm{obs}},\vartheta,M)
(10)
Rearranging the marginalization integral leads to the following interpretable form.
p
⁡
(
M
|
Z
,
𝒴
obs
)
\displaystyle p(M|Z,\mathcal{Y}^{\mathrm{obs}})
∝
∫
d
​
𝒴
​
[
{
∫
d
​
F
​
p
​
(
Z
|
F
,
𝒴
)
​
p
​
(
F
)
}
​
{
∫
d
​
ϑ
​
p
​
(
𝒴
|
ϑ
)
​
p
​
(
𝒴
obs
|
ϑ
)
​
p
​
(
ϑ
|
M
)
}
​
p
​
(
M
)
]
\displaystyle\propto\int d\mathcal{Y}\,\left[\left\{\int dF\>p(Z|F,\mathcal{Y})p(F)\right\}\left\{{{\int d\vartheta\,p(\mathcal{Y}|\vartheta){\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta)p(\vartheta|M)}}\right\}p(M)\right]
=
∫
d
​
𝒴
​
p
​
(
Z
|
𝒴
)
​
{
∫
d
​
ϑ
​
p
​
(
𝒴
|
ϑ
)
​
p
​
(
ϑ
|
𝒴
obs
,
M
)
}
​
p
​
(
𝒴
obs
|
M
)
​
p
​
(
M
)
\displaystyle=\int d\mathcal{Y}\>p(Z|\mathcal{Y})\left\{{{\int d\vartheta\,p(\mathcal{Y}|\vartheta){\it p}(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)}}\right\}{\it p}(\mathcal{Y}^{\mathrm{obs}}|M)p(M)
(11)
=
p
⁡
(
𝒴
obs
|
M
)
​
p
​
(
M
)
​
∫
d
​
𝒴
​
p
​
(
Z
|
𝒴
)
​
p
​
(
𝒴
|
𝒴
obs
,
M
)
\displaystyle={\it p}(\mathcal{Y}^{\mathrm{obs}}|M)p(M)\int d\mathcal{Y}\>p(Z|\mathcal{Y})p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)
(12)
=
p
⁡
(
𝒴
obs
|
M
)
​
p
​
(
M
)
​
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
,
\displaystyle=p(\mathcal{Y}^{\mathrm{obs}}|M)p(M)E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right],
(13)
where, the
ϑ
\vartheta
integral marginalizes the spectral deconvolution model, the
F
F
integral marginalizes the Gaussian process regression model, and the
𝒴
\mathcal{Y}
integral marginalizes over the connection between layers in the hierarchical Bayesian spectral deconvolution model for scientists.
Each marginalization will be explained in the next section.
Assuming a uniform prior of
p
⁡
(
M
)
p(M)
over candidate models, model comparison based on
p
⁡
(
M
|
Z
,
𝒴
obs
)
p(M|Z,\mathcal{Y}^{\mathrm{obs}})
is equivalent to comparison based on the model evidence
p
⁡
(
Z
,
𝒴
obs
|
M
)
p(Z,\mathcal{Y}^{\mathrm{obs}}|M)
.
Equation (
13
) shows that the full Bayesian model evidence can be decomposed into two conceptually distinct contributions:
p
⁡
(
Z
,
𝒴
obs
|
M
)
=
p
⁡
(
𝒴
obs
|
M
)
​
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
.
p(Z,\mathcal{Y}^{\mathrm{obs}}|M)=p(\mathcal{Y}^{\mathrm{obs}}|M)E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right].
(14)
The first contribution is the spectrum-only evidence
p
⁡
(
𝒴
obs
|
M
)
p(\mathcal{Y}^{\mathrm{obs}}|M)
, which evaluates how well the candidate spectral model
M
M
explains the observed spectra themselves.
The second contribution is the physical-property consistency term,
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
​
(
Z
|
𝒴
)
]
,
E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right],
(15)
which evaluates whether the latent spectral structures inferred under model
M
M
, after conditioning on the observed spectra, can consistently explain the physical-property values.
This distinction becomes explicit by taking the negative logarithm of the model evidence.
The full Bayesian free energy is defined as
F
​
E
full
​
(
M
)
=
−
log
⁡
p
⁡
(
Z
,
𝒴
obs
|
M
)
.
FE_{\mathrm{full}}(M)=-\log p(Z,\mathcal{Y}^{\mathrm{obs}}|M).
(16)
Using the decomposition above, this free energy can be written as
F
​
E
full
​
(
M
)
=
F
​
E
spec
​
(
M
)
+
F
​
E
phys
​
(
M
)
,
FE_{\mathrm{full}}(M)=FE_{\mathrm{spec}}(M)+FE_{\mathrm{phys}}(M),
(17)
where
F
​
E
spec
​
(
M
)
=
−
log
⁡
p
⁡
(
𝒴
obs
|
M
)
FE_{\mathrm{spec}}(M)=-\log p(\mathcal{Y}^{\mathrm{obs}}|M)
(18)
is the spectrum-only Bayesian free energy, and
F
​
E
phys
​
(
M
)
=
−
log
⁡
p
⁡
(
Z
|
𝒴
obs
,
M
)
=
−
log
⁡
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
FE_{\mathrm{phys}}(M)=-\log p(Z|\mathcal{Y}^{\mathrm{obs}},M)=-\log E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right]
(19)
is the physical-property-consistency free energy.
These two terms evaluate different aspects of the candidate model:
F
​
E
spec
FE_{\mathrm{spec}}
measures the fit to the observed spectra, whereas
F
​
E
phys
FE_{\mathrm{phys}}
measures the consistency between the inferred spectral structures and the physical-property values.
The purpose of this study is not to select the spectral model that best fits the observed spectra alone, but to select the model whose inferred spectral structures are most informative for the physical-property values.
Therefore, in the proposed method, we use the physical-property-consistency term as the model-selection criterion.
We define the physical-property-informed model posterior as
p
~
​
(
M
|
Z
,
𝒴
obs
)
∝
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
.
\tilde{p}(M|Z,\mathcal{Y}^{\mathrm{obs}})\propto E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right].
(20)
The corresponding physical-property-informed free energy is defined as
F
​
E
~
​
(
M
)
=
−
log
⁡
p
~
​
(
M
|
Z
,
𝒴
obs
)
.
\widetilde{FE}(M)=-\log\tilde{p}(M|Z,\mathcal{Y}^{\mathrm{obs}}).
(21)
In the proposed method, the candidate spectral model with the smallest
F
​
E
~
​
(
M
)
\widetilde{FE}(M)
is selected.
II.3
Posterior inference for the model
Estimating the physical-property-informed model posterior
p
~
​
(
M
|
Z
,
𝒴
obs
)
\tilde{p}(M|Z,\mathcal{Y}^{\mathrm{obs}})
for the scientific spectral model, which encodes the physical properties, requires evaluating the marginalization integrals over
θ
\theta
,
F
F
, and
𝒴
\mathcal{Y}
, as detailed in Eq. (
13
).
These operations correspond to the marginalization within the spectral deconvolution model, Gaussian process regression model, and interface linking the two layers.
Marginalization proceeds sequentially over
ϑ
\vartheta
,
F
F
, and
𝒴
\mathcal{Y}
.
This section presents the method for each step in this sequence.
II.3.1
Marginalization of spectral deconvolution model
The method for sampling from the posterior of the spectral deconvolution model using Replica Exchange Monte Carlo is described here.
The purpose of the marginalization calculation for the spectral deconvolution model,
∫
d
​
θ
​
p
​
(
𝒴
|
ϑ
)
​
p
​
(
𝒴
obs
|
ϑ
)
​
p
​
(
ϑ
|
M
)
\int d\theta\,p(\mathcal{Y}|\vartheta){\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta)\,p(\vartheta|M)
, is to compute a sample set from the predictive distribution
p
⁡
(
𝒴
|
𝒴
obs
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}})
.
Sampling from this predictive distribution
p
⁡
(
𝒴
|
𝒴
obs
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}})
can be achieved using the following procedure, as seen in Eq. (
11
).
The method proceeds by first drawing the parameters
ϑ
\vartheta
from the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
. Then, sampling
𝒴
\mathcal{Y}
from the spectral deconvolution likelihood
p
⁡
(
𝒴
|
ϑ
)
p(\mathcal{Y}|\vartheta)
(Eq. (
2
)) conditioned on these values yields the sample set from the predictive distribution
p
⁡
(
𝒴
|
𝒴
obs
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}})
.
Because the likelihood function
p
⁡
(
𝒴
|
ϑ
)
p(\mathcal{Y}|\vartheta)
is Gaussian, sampling from it is numerically straightforward. Therefore, the primary computational challenge lies in sampling from the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
.
Sampling from posterior distributions in high-dimensional statistical models is often performed using Markov chain Monte Carlo methods. Because posterior distributions in spectral deconvolution can be strongly multimodal, standard Metropolis–Hastings sampling
13
may require many iterations to move between modes and can converge slowly
35
;
23
.
This study employed the Replica Exchange Monte Carlo (REMC) method, an extended-sampling method
15
, to address these sampling challenges.
Constructing an extended artificial ensemble and exchanging states can efficiently enhance Markov chain mixing using extended-sampling methods
15
.
We employed an established approach
17
that marginalizes the spectral model while avoiding this difficulty.
Sampling from the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
for the spectral deconvolution model does not require the simultaneous use of all spectral data.
The likelihood function
p
⁡
(
𝒴
obs
|
ϑ
)
p(\mathcal{Y}^{\mathrm{obs}}|\vartheta)
factorizes into the individual spectral likelihoods
p
⁡
(
Y
j
obs
|
θ
)
p({Y_{j}}^{\mathrm{obs}}|\theta)
, as shown in Eq. (
8
).
By Bayes’ theorem, the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
is proportional to the likelihood function
p
⁡
(
𝒴
obs
|
ϑ
)
​
p
​
(
ϑ
|
M
)
{\it p}(\mathcal{Y}^{\mathrm{obs}}|\vartheta)p(\vartheta|M)
. Therefore, the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
also corresponds to the product of the posterior distributions
p
⁡
(
θ
|
Y
j
obs
)
{\it p}(\theta|{Y_{j}}^{\mathrm{obs}})
for individual spectra.
Sampling from the posterior distribution
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
can be achieved by sampling from the posterior distribution of the individual spectra
p
⁡
(
θ
|
Y
j
obs
)
{\it p}(\theta|{Y_{j}}^{\mathrm{obs}})
.
Hence, we describe a method for sampling from the posterior distribution
p
⁡
(
θ
|
Y
j
obs
)
{\it p}(\theta|{Y_{j}}^{\mathrm{obs}})
of a single spectral data point
Y
obs
:=
Y
j
obs
Y^{\mathrm{obs}}:=Y_{j}^{\mathrm{obs}}
using the replica exchange method.
Building on this, we define the probability distribution
p
⁡
(
θ
|
Y
obs
,
M
)
p(\theta|Y^{\mathrm{obs}},M)
by introducing the inverse temperature parameter
β
\beta
into the probability distribution
p
β
​
(
θ
|
Y
obs
,
M
)
p_{\beta}(\theta|Y^{\mathrm{obs}},M)
of the parameter set.
p
β
​
(
θ
|
Y
obs
,
M
)
p_{\beta}(\theta|Y^{\mathrm{obs}},M)
is calculated as
p
β
​
(
θ
|
Y
obs
,
M
)
∝
exp
⁡
(
−
N
​
β
​
E
​
(
θ
,
M
)
)
​
p
​
(
θ
)
,
\displaystyle p_{\beta}(\theta|Y^{\mathrm{obs}},M)\propto\exp(-N\beta E(\theta,M))p(\theta),
(22)
E
⁡
(
θ
,
M
)
:=
1
2
​
N
​
σ
2
​
∑
i
=
1
N
(
y
i
−
g
⁡
(
x
i
,
θ
,
M
)
)
2
.
\displaystyle E(\theta,M):=\frac{1}{2N\sigma^{2}}\sum_{i=1}^{N}(y_{i}-g(x_{i};\theta,M))^{2}.
(23)
As can be seen from this equation, when
β
=
1
\beta=1
, the sampling matches the target distribution
p
⁡
(
θ
|
Y
obs
)
p(\theta|Y^{\mathrm{obs}})
.
However, reducing
β
\beta
corresponds to sampling from the prior distribution as the distribution approaches it.
REMC is an efficient method for simultaneous sampling from different temperatures
β
l
\beta_{l}
, enabling the simultaneous distribution of the parameter values
{
θ
l
}
l
=
1
L
\{\theta_{l}\}_{l=1}^{L}
corresponding to
β
l
\beta_{l}
to be obtained.
P
(
θ
1
,
θ
2
⋯
θ
L
|
Y
obs
,
M
)
=
∏
l
=
1
L
P
β
l
(
θ
l
|
Y
obs
,
M
)
{\it P}(\theta_{1},\theta_{2}\cdots\theta_{L}|Y^{\mathrm{obs}},M)=\prod_{l=1}^{L}{\it P}_{\beta_{l}}(\theta_{l}|Y^{\mathrm{obs}},M)
(24)
is sampled by repeating the following procedure.
1
Sampling from the individual distribution
p
⁡
(
θ
l
|
Y
obs
,
β
l
)
{\it p}(\theta_{l}|Y^{\mathrm{obs}},\beta_{l})
By using sampling methods such as the Metropolis–Hastings algorithm
13
, we sample the parameter
θ
l
\theta_{l}
from
p
β
l
​
(
θ
l
|
Y
obs
,
M
)
{\it p}_{\beta_{l}}(\theta_{l}|Y^{\mathrm{obs}},M)
.
2
Probabilistically swap sampling at each
β
\beta
Swap
θ
l
\theta_{l}
and
θ
l
+
1
\theta_{l+1}
with the probability
min
⁡
(
1
,
r
)
\min(1,r)
.
r
=
p
(
θ
1
,
⋯
,
θ
l
+
1
,
θ
l
,
⋯
,
θ
L
|
Y
obs
,
M
)
p
(
θ
1
,
⋯
,
θ
l
,
θ
l
+
1
,
⋯
,
θ
L
|
Y
obs
,
M
)
=
p
β
l
​
(
θ
l
+
1
|
Y
obs
,
M
)
​
p
β
l
+
1
​
(
θ
l
|
Y
obs
,
M
)
p
β
l
​
(
θ
l
|
Y
obs
,
M
)
​
p
β
l
+
1
​
(
θ
l
+
1
|
Y
obs
,
M
)
=
exp
⁡
{
N
⁡
[
β
l
+
1
−
β
l
]
​
[
E
⁡
(
θ
l
+
1
,
M
)
−
E
⁡
(
θ
l
,
M
)
]
}
\begin{split}\;\;\;\;\;r&=\frac{{\it p}(\theta_{1},\cdots,\theta_{l+1},\theta_{l},\cdots,\theta_{L}|Y^{\mathrm{obs}},M)}{{\it p}(\theta_{1},\cdots,\theta_{l},\theta_{l+1},\cdots,\theta_{L}|Y^{\mathrm{obs}},M)}\\
&=\frac{{\it p}_{\beta_{l}}(\theta_{l+1}|Y^{\mathrm{obs}},M){\it p}_{\beta_{l+1}}(\theta_{l}|Y^{\mathrm{obs}},M)}{{\it p}_{\beta_{l}}(\theta_{l}|Y^{\mathrm{obs}},M){\it p}_{\beta_{l+1}}(\theta_{l+1}|Y^{\mathrm{obs}},M)}\\
&=\exp\left\{N[\beta_{l+1}-\beta_{l}][E(\theta_{l+1},M)-E(\theta_{l},M)]\right\}\end{split}
(25)
Exchanging the states across multiple inverse temperatures
β
\beta
helps the sampling to converge faster for
p
⁡
(
θ
|
𝒴
obs
)
p(\theta|\mathcal{Y}^{\mathrm{obs}})
at
β
=
1
\beta=1
.
Combining the samples from these spectral posteriors
p
⁡
(
θ
|
𝒴
obs
,
M
)
p(\theta|\mathcal{Y}^{\mathrm{obs}},M)
creates the global posterior set
p
⁡
(
ϑ
|
𝒴
obs
,
M
)
p(\vartheta|\mathcal{Y}^{\mathrm{obs}},M)
.
Finally, drawing
𝒴
\mathcal{Y}
from the Gaussian likelihood
p
⁡
(
𝒴
|
ϑ
)
p(\mathcal{Y}|\vartheta)
using these parameters gives the final sample set
{
𝒴
k
}
k
=
1
N
s
​
a
​
m
​
p
\{\mathcal{Y}_{k}\}_{k=1}^{N_{samp}}
. This set forms the spectral predictive distribution
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)
with
N
s
​
a
​
m
​
p
N_{samp}
samples.
II.3.2
Marginalization of Physical-Property Regression Models
This section describes the marginalization calculation for the physical-property regression model defined in Sec.
II.2.2
.
The spectral samples
{
Y
k
}
k
=
1
N
samp
\{Y_{k}\}_{k=1}^{N_{\mathrm{samp}}}
drawn from
p
⁡
(
Y
|
ϑ
)
p(Y|\vartheta)
are combined with the measurement data
Z
Z
to form the dataset
{
𝒴
k
,
Z
}
k
=
1
N
s
​
a
​
m
​
p
\{\mathcal{Y}_{k},Z\}_{k=1}^{N_{samp}}
.
N
s
​
a
​
m
​
p
N_{samp}
denotes the sample size drawn from the predictive distribution
p
⁡
(
𝒴
|
ϑ
)
p(\mathcal{Y}|\vartheta)
.
While performing this calculation, it is important to note that the physical property
Z
Z
does not change across the samples
k
k
.
For each sampled set
{
𝒴
k
,
Z
}
\left\{\mathcal{Y}_{k},Z\right\}
, the marginal likelihood
p
⁡
(
Z
|
𝒴
k
)
p(Z|\mathcal{Y}_{k})
can be analytically computed, as follows.
Denoting the
N
′
×
N
′
N^{\prime}\times N^{\prime}
identity matrix as
I
N
′
I_{N^{\prime}}
and defining
Λ
=
K
+
α
−
1
​
I
N
′
\Lambda=K+\alpha^{-1}I_{N^{\prime}}
, Gaussian process regression yields the conditional probability
p
⁡
(
Z
|
𝒴
k
)
p(Z|{\mathcal{Y}}_{k})
after integrating out
F
F
, as shown in the following equation.
p
⁡
(
Z
|
𝒴
k
)
=
∫
d
​
F
​
p
​
(
Z
|
F
,
𝒴
k
)
​
p
​
(
F
)
\displaystyle p(Z|{\mathcal{Y}}_{k})=\int dFp(Z|F,{\mathcal{Y}}_{k})p(F)
(26)
=
\displaystyle=
(
1
2
​
π
)
N
′
/
2
​
|
Λ
−
1
|
1
/
2
​
exp
⁡
{
−
1
2
​
Z
T
​
Λ
−
1
​
Z
}
\displaystyle\left(\frac{1}{2\pi}\right)^{N^{\prime}/2}|\Lambda^{-1}|^{1/2}\exp\left\{-\frac{1}{2}Z^{T}\Lambda^{-1}Z\right\}
Thus, the set of
N
s
​
a
​
m
​
p
N_{samp}
marginalized likelihoods
p
⁡
(
Z
|
𝒴
k
)
p(Z|\mathcal{Y}_{k})
, denoted as
{
p
⁡
(
Z
|
𝒴
k
)
}
k
=
1
N
s
​
a
​
m
​
p
\{p(Z|\mathcal{Y}_{k})\}_{k=1}^{N_{samp}}
, is obtained.
II.3.3
Marginalization of models of spectral deconvolution processes used by scientists
Marginalization over
𝒴
\mathcal{Y}
means taking the expectation (average value) of
p
⁡
(
Z
|
𝒴
)
p(Z|\mathcal{Y})
with respect to the probability distribution
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)
, where
𝒴
\mathcal{Y}
is the set of all possible values,
𝒴
obs
\mathcal{Y}^{\mathrm{obs}}
represents the observed values, and
M
M
denotes the model. This is shown in the following equation.
p
~
​
(
M
|
Z
,
𝒴
obs
)
\displaystyle\tilde{p}(M|Z,\mathcal{Y}^{\mathrm{obs}})
∝
∫
d
​
𝒴
​
p
​
(
Z
|
𝒴
)
​
p
​
(
𝒴
|
𝒴
obs
,
M
)
\displaystyle\propto\int d\mathcal{Y}\ p(Z|\mathcal{Y})p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)
=
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
​
(
Z
|
𝒴
)
]
\displaystyle=E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right]
(27)
The expectation in Eq. (
II.3.3
) was evaluated by Monte Carlo averaging over the sampling set
{
p
⁡
(
Z
|
𝒴
k
)
}
k
=
1
N
samp
\{p(Z|\mathcal{Y}_{k})\}_{k=1}^{N_{\mathrm{samp}}}
.
Here, each
𝒴
k
\mathcal{Y}_{k}
was sampled from the predictive distribution
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)
,
which was generated using the sampling sequence of the parameter set
ϑ
\vartheta
obtained from the spectral deconvolution model.
Thus, the physical-property-informed model posterior was approximated as
p
~
​
(
M
|
Z
,
𝒴
obs
)
∝
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
≈
1
N
samp
​
∑
k
=
1
N
samp
p
⁡
(
Z
|
𝒴
k
)
.
\tilde{p}(M|Z,\mathcal{Y}^{\mathrm{obs}})\propto E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right]\approx\frac{1}{N_{\mathrm{samp}}}\sum_{k=1}^{N_{\mathrm{samp}}}p(Z|\mathcal{Y}_{k}).
(28)
Building on this approximation, computing the physical-property-informed free energy (
FE
~
\widetilde{\rm FE}
) as
F
​
E
~
​
(
M
)
=
−
log
⁡
E
p
⁡
(
𝒴
|
𝒴
obs
,
M
)
​
[
p
⁡
(
Z
|
𝒴
)
]
≈
−
log
⁡
[
1
N
samp
​
∑
k
=
1
N
samp
p
⁡
(
Z
|
𝒴
k
)
]
\widetilde{FE}(M)=-\log E_{p(\mathcal{Y}|\mathcal{Y}^{\mathrm{obs}},M)}\left[p(Z|\mathcal{Y})\right]\approx-\log\left[\frac{1}{N_{\mathrm{samp}}}\sum_{k=1}^{N_{\mathrm{samp}}}p(Z|\mathcal{Y}_{k})\right]
enables spectral model selection.
