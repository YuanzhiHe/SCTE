# TopoAgent: An Agentic Framework for Automated Topology Learning in Medical Imaging

paper_id: arxiv:2606.29763v1
tier: H
source_used: html_arxiv
warning: none

## Intro

Persistent homology (PH), a central tool in topological data analysis (TDA), captures geometric structural properties in images that conventional pixel-level deep learning (DL) approaches often neglect
[
16
,
10
,
42
,
36
,
1
]
. In medical imaging, this topological perspective has proven valuable for grading cancer
[
33
,
58
]
, predicting breast cancer survival
[
51
]
, analyzing retinal vasculature
[
4
]
, and classifying tissue morphology
[
24
,
23
]
, with growing adoption in other visual domains such as shape analysis and materials science
[
53
]
. To make these topological insights actionable for downstream tasks, researchers rely on
topological descriptors
: methods that convert persistence diagrams (PDs) or raw images into fixed-length topological feature vectors
[
5
,
53
]
. A rich family of such descriptors is known, including persistence images
[
2
]
, persistence
landscape
[
9
]
, Euler characteristic transform
[
55
]
, and Minkowski functional
[
41
]
, among others (see Supplementary for the complete descriptor taxonomy). However, each descriptor encodes distinct geometric characteristics and entails specific mathematical properties, hyperparameters, and failure modes. No single descriptor is universally effective across all datasets
[
5
]
(e.g., see Fig.
1
(a)). Thus, determining effective descriptors for a given image dataset is an overwhelming challenge that demands a great deal of expertise and effort.
(a) Accuracy of different topological descriptors on five datasets
Dataset
PS
[
5
]
PI
[
2
]
TF
[
47
]
ATOL
[
49
]
MK
[
41
]
LBP
[
45
]
Best
BloodMNIST
[
64
]
96.75
90.63
97.92
97.64
97.75
90.03
TF
BreakHis
[
52
]
88.76
67.92
79.64
87.03
83.91
77.02
PS
BrainTumor
[
14
]
95.10
83.49
95.12
96.16
94.71
94.98
ATOL
ISIC2019
[
15
]
36.14
28.17
33.31
34.05
36.94
29.66
MK
APTOS2019
[
7
]
55.03
39.65
53.84
51.83
52.06
57.74
LBP
(b) Reasoning comparison on a BreakHis image
Input
Prompt:
“Determine the best topological descriptor.”
Input Image:
MedRAX
[
18
]
Analysis:
“
H
1
→
H_{1}\rightarrow
glandular lumens. Signal is
noisy
.”
Decision:
Better medical interpretation, but same noise-driven default.
Result:
Pers. Silhouette
×
Claude
[
6
]
Analysis:
“
H
1
H_{1}
reflects loops from lumens and keratin whorls.”
Decision:
Richest anatomy; no empirical evidence to verify.
Result:
Pers. Landscape
×
Gemini
[
20
]
Analysis:
“Low avg persistence
→
\rightarrow
noise.
H
1
H_{1}
max
=
{=}
0.38
→
\rightarrow
stable loops.”
Decision:
Sound TDA reasoning, but produces invalid tool parameters.
Result:
Pers. Image
×
TopoAgent
Perception:
glands/lumens
; balanced
H
0
H_{0}
/
H
1
H_{1}
.
Reasoning:
Propose Pers. Image.
Rankings
: Pers. Statistics is Tier 1
→
\rightarrow
switch.
Memory
: confirmed (3 trials).
Action:
Call Pers. Statistics.
Reflection:
0% sparsity, high variance.
Result:
Pers. Statistics
✓
\checkmark
Figure 1
:
(a) Balanced accuracy (%) of 6 descriptors (covering the top-3 per object type) on 5 datasets. PS = persistence statistics, PI = persistence image, TF = template function, ATOL = automatic topologically-oriented learning, MK = Minkowski functional, LBP = local binary pattern. The best descriptor (bold) differs across 5 datasets, confirming no single descriptor is best for all.
(b)
Reasoning comparison on a BreakHis image (Pers. = persistence). General-purpose LLMs with the same tools all select suboptimal descriptors; GPT-4o
[
28
]
(MedRAX’s backbone) reaches similar result without medical insight. TopoAgent determines persistence statistics for glands/lumens.
This determination is challenging because it depends on the interaction between dataset-specific morphological characteristics (e.g., connected components, loops, cavities) and each descriptor’s mathematical properties. In practice, users must manually analyze dataset properties, choose among candidate descriptors, tune hyperparameters, and produce topological feature vectors for downstream tasks. With a large descriptor library and continuous parameter spaces, exhaustive evaluation is prohibitive at the per-image level, making this trial-and-error process time-consuming, expertise-dependent, and difficult to standardize.
Large language model (LLM)-based agentic frameworks offer a “natural” solution. Modern agents integrate reasoning, tool use, memory, and iterative refinement
[
65
,
50
,
70
,
71
]
, and medical AI agents have begun orchestrating domain-specific tools for clinical analysis
[
18
,
35
,
60
,
63
,
62
]
. Since descriptor determination requires both high-level reasoning about data morphology and low-level computation (e.g., PH and descriptor extraction), it aligns naturally with this paradigm. However, no existing agentic framework automatically computes PH, reasons about topological structures, and produces topological feature vectors for downstream tasks. General-purpose LLMs can describe topology in natural language but cannot compute it without external tool integration
[
38
]
, and even with tools, they lack the empirical grounding to verify their determinations (e.g., see Fig.
1
(b)).
Automating descriptor determination also requires systematic evaluation, but prior studies typically assess a small number of descriptors on limited datasets with varying preprocessing, classifiers, and evaluation metrics
[
5
]
, making cross-study comparisons unreliable. No standardized benchmark spans diverse medical image morphologies with consistent criteria, leaving the community without clear guidance on when specific topological descriptors excel or fail.
To address these gaps, we propose
TopoAgent
and
TopoBenchmark
. TopoAgent is an LLM-based agentic framework that automates topology learning for medical images by determining the most suitable topological descriptors for an input dataset. Given a raw image and a task prompt, TopoAgent operates through a Perception–Reasoning–Action–Reflection (PRAR) loop without task-specific training. Perception characterizes the image’s topological and visual properties using perception tools. Reasoning first formulates a descriptor proposal by matching the observed characteristics based on descriptor properties, and then integrates empirical evidence from a distilled skill set and dual memory to determine the final descriptor and parameters.
Action executes the determined descriptor using descriptor tools to produce a topological feature vector. Reflection validates the output quality and, on failure, records a diagnosis in long-term memory and triggers retry, enabling the agent to self-correct rather than committing to a single-shot decision. TopoBenchmark is a standardized evaluation benchmark comprising 26 datasets that broadly cover publicly available 2D medical imaging scenes, organized into five object types that span the major morphological categories in medical images:
cells
,
glands and lumens
,
organ shapes
,
vessel trees
, and
surface lesions
. It provides an empirical basis from which the skill set is distilled and a frozen testbed of 113,182 samples with convergence-based per-dataset sample sizing for consistent evaluation.
Our contributions are threefold. (1) We introduce TopoAgent, the first agentic framework that bridges LLM-based reasoning and topological data analysis, enabling automated topology learning from raw medical images with full reasoning traces. (2) We construct TopoBenchmark, the first standardized benchmark for evaluating topological descriptors across diverse medical image morphologies with consistent criteria. (3) We demonstrate that adaptive, per-image descriptor determination outperforms the strongest baseline by 9.32% and general-purpose LLMs equipped with the same tools by over 21% in average balanced accuracy. Our code is publicly available at
https://github.com/gm3g11/TopoAgent
.

## Method

Figure 2
:
An overview of the TopoAgent framework with four phases:
Perception
identifies the object type and analyzes persistent homology statistics;
Reasoning
proposes and
determines the descriptor and parameters;
Action
calls the descriptor tools;
Reflection
validates the feature vector. The skill set
𝒮
\mathcal{S}
feeds descriptor properties and tiered rankings with parameters to Reasoning (dashed green). Short-term memory
M
s
M_{s}
tracks execution history within a run; long-term memory
M
l
M_{l}
supplies diagnostic entries to Reasoning (dashed orange) and is updated on failure.
3.1
Framework Overview
Given a medical image
I
I
and a task prompt, TopoAgent determines the most suitable topological descriptor
d
∗
∈
𝒟
d^{*}\in\mathcal{D}
from a library of
|
𝒟
|
=
15
|\mathcal{D}|=15
descriptors, configures its parameters
θ
∗
\theta^{*}
, and produces a topological feature vector
𝐟
∈
ℝ
n
d
\mathbf{f}\in\mathbb{R}^{n_{d}}
along with a reasoning trace
ℛ
\mathcal{R}
documenting all decisions.
It uses the PRAR architecture
[
59
,
61
,
30
]
, extended with tools, a distilled skill set
𝒮
\mathcal{S}
, and dual memory (
M
s
M_{s}
,
M
l
M_{l}
). The skill set
𝒮
\mathcal{S}
comprises three components: descriptor properties
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
encoding qualitative knowledge, empirical evidence
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
providing tiered rankings and reasoning chains, and validated parameters
𝒮
param
\mathcal{S}_{\mathrm{param}}
specifying optimal configurations per object type. Short-term memory
M
s
M_{s}
records tool invocations within each run; long-term memory
M
l
M_{l}
keeps diagnostic entries across runs.
Fig.
2
shows the
workflow.
We present the four-phase pipeline in §
3.2
and
the supporting components in §
3.3
.
3.2
TopoAgent Pipeline
Perception.
Before reasoning about descriptor suitability, the agent must first understand the image: its imaging modality, tissue or anatomical structures, and topological characteristics. Perception invokes six perception tools deterministically, producing three outputs. (a) The PH profile
h
⁡
(
I
)
h(I)
summarizes the topological characteristics: birth-death pair counts per homology dimension, average persistence, Betti ratio
β
1
/
β
0
\beta_{1}/\beta_{0}
(the ratio of loop count to component count), and per-dimension persistence distributions. (b) Visual statistics
v
⁡
(
I
)
v(I)
capture image-level properties such as signal-to-noise ratio, contrast, and edge density. (c) The object type
o
∈
𝒪
=
{
cells
,
glands/lumens
,
organ shapes
,
vessel trees
,
surface lesions
}
o\in\mathcal{O}=\{\textit{cells},\textit{glands/lumens},\textit{organ shapes},\textit{vessel trees},\textit{surface lesions}\}
is identified from the image content. Deterministic tool execution ensures reproducible PH computation. The LLM’s role in Perception is interpretive: it jointly reasons on the raw image and the computed PH to identify
o
o
. The object type
o
o
is critical because it indexes the skill set: empirical evidence
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
and validated parameters
𝒮
param
\mathcal{S}_{\mathrm{param}}
are all organized per object type.
Both topological and visual
information are necessary for reliable object-type identification. The PH profile quantifies topological characteristics that directly inform descriptor choice, such as the relative prevalence of connected components versus loops and their persistence distributions. However, the PH profile alone cannot reliably distinguish all object types. For example, both dense cell populations and glandular tissues with fragmented lumen boundaries can give somewhat similar
H
0
H_{0}
-dominated PH profiles, but they require fundamentally different descriptors due to distinct underlying morphology. Visual grounding resolves such ambiguities: jointly observing the image and its PH substantially outperforms both PH-profile-only and vision-only identification (Table
2
), confirming that the two information sources provide complementary evidence.
Reasoning.
Reasoning is the central phase of the pipeline, responsible for both proposing and determining the descriptor configuration. It addresses the core challenge of the framework:
determining an effective topological descriptor requires understanding of the interaction between the image’s topological characteristics and each descriptor’s mathematical properties. To handle this, Reasoning
takes two steps with asymmetric information access.
In the
proposal step
, the agent receives the Perception outputs
(
h
⁡
(
I
)
,
v
⁡
(
I
)
,
o
)
(h(I),v(I),o)
together with
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
, which encodes the mathematical definition, strengths, weaknesses, and intended use of each descriptor. Crucially,
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
is provided in stripped form: the agent sees reasoning patterns that describe how PH characteristics relate to descriptor suitability, but not the descriptor recommendations or tiered rankings themselves. The agent reasons about which descriptor’s mathematical properties best match the observed topological characteristics and proposes a candidate
(
d
p
,
θ
p
)
(d_{p},\theta_{p})
with preliminary parameters. In the
determination step
, the agent integrates the full
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
(including tiered rankings, descriptor recommendations, and threshold-based PH signal definitions) and long-term memory
M
l
M_{l}
(outcomes from prior runs on the current dataset). Weighing the proposal against this empirical evidence, the agent arrives at one of three decisions: it retains
(
d
p
,
θ
p
)
(d_{p},\theta_{p})
when the PH profile provides strong evidence despite a low ranking, defers to
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
when the proposal is weakly supported, or adopts a correction from
M
l
M_{l}
that overrides both.
This two-step design addresses anchoring bias, the tendency of LLMs to fixate on the first salient information that they receive rather than reasoning independently. When the full
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
is visible during proposal formation, the LLM determines the top-ranked descriptor in 92.1% of 520 held-out test images (20 per dataset in 26 datasets), even when explicitly prompted to “refer to but not rely on” the rankings. This reduces the agent into a lookup table. Providing only stripped reasoning patterns during the proposal step reduces the top-ranked agreement rate to 61.3%: of these, 38.8% are cases where the agent independently arrived at the same determination, while 22.5% are cases where
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
corrected the proposal during determination (more details in the Supplementary).
Action.
Action executes the determined descriptor
d
∗
d^{*}
with parameters
θ
∗
\theta^{*}
by invoking the corresponding descriptor tools, producing a topological feature vector
𝐟
∈
ℝ
n
d
\mathbf{f}\in\mathbb{R}^{n_{d}}
. The parameters are drawn from the ranges validated during skill set construction for the identified object type
o
o
rather than generated by the LLM, ensuring reproducible and correctly configured tool calls.
Reflection.
A single pass through Reasoning and Action does not guarantee a suitable output: the determined descriptor may produce an unsuitable feature vector due to parameter misconfiguration (e.g., overly fine resolution causing excessive sparsity), a mismatch with the image’s topological characteristics, or edge-case PH characteristics not anticipated by
𝒮
\mathcal{S}
. Reflection addresses this by having the LLM assess the quality of
𝐟
\mathbf{f}
based on summary statistics (sparsity, variance, kurtosis, skewness, dynamic range, and informative-feature ratio) together with descriptor-specific reference ranges from
𝒮
\mathcal{S}
and diagnostic entries from
M
l
M_{l}
. Crucially, reference ranges are provided as context, not as hard thresholds: what constitutes acceptable quality varies across descriptors and tissue types. For instance, high sparsity in a persistence image may be appropriate for sparse PH but indicates failure on dense glandular tissue; LBP histograms naturally exhibit high kurtosis that fixed thresholds would flag as anomalous. If the LLM diagnoses a quality failure, it returns a structured correction (e.g., “switch to a lower-dimensional descriptor”), recorded in
M
l
M_{l}
along with the failed descriptor and diagnosed cause, then retries from the determination step of Reasoning. A maximum of two retries is permitted within
t
max
t_{\max}
; if no passing result is obtained, the agent falls back to the top-ranked descriptor for
o
o
from
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
. Reflection validates feature vector quality, not downstream classification accuracy, since class labels are unavailable at inference time. On success, TopoAgent returns
𝐟
\mathbf{f}
together with a reasoning trace
ℛ
=
{
h
⁡
(
I
)
,
v
⁡
(
I
)
,
o
,
d
p
,
d
∗
,
θ
∗
,
q
}
\mathcal{R}=\{h(I),v(I),o,d_{p},d^{*},\theta^{*},q\}
, where
q
q
denotes the quality diagnosis.
3.3
TopoAgent Components
Toolset.
The agent accesses 21 domain-specific tools organized into two groups. 6
perception tools
support the Perception phase: image loading, quality analysis, denoising, PH computation via GUDHI
[
40
]
with cubical complex filtration, PH profile extraction, and Betti ratio computation. 15
descriptor tools
support the Action phase: 10 PH-derived descriptors that vectorize persistence diagrams (e.g., persistence image
[
2
]
, persistence landscape
[
9
]
, ATOL
[
49
]
) and 5 image-based descriptors that capture geometric properties without explicit PH computation (e.g., Minkowski functional
[
41
]
, Euler characteristic transform
[
55
]
, LBP texture
[
45
]
); see Supplementary for the complete taxonomy. Each tool exposes configurable parameters whose optimal values per object type are determined during skill set construction. Implementing descriptors as pre-built, parameterized tools ensures that the LLM reasons about descriptor suitability rather than generating computation code.
Skill Set.
To determine suitable descriptors, TopoAgent needs domain expertise that LLMs lack: understanding of each descriptor’s mathematical properties, empirical evidence of descriptor effectiveness per object type, and appropriate parameter configurations. We encode this expertise in a static skill set
𝒮
\mathcal{S}
, distilled offline from systematic evaluation on TopoBenchmark (§
3.4
): descriptor properties
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
, empirical evidence
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
, and validated parameters
𝒮
param
\mathcal{S}_{\mathrm{param}}
.
As shown in Fig.
3
,
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
encodes qualitative knowledge for the proposal step: the mathematical definition, strengths, weaknesses, and intended use of each descriptor, along with parameter reasoning heuristics that guide the LLM toward appropriate configurations given the observed PH profile.
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
provides empirical evidence for the determination step: per-object-type descriptor rankings presented as tiers rather than exact accuracy values, reasoning chains that map PH profile patterns to descriptor recommendations, and threshold-based PH signal definitions. As detailed in §
3.2
, the proposal step receives reasoning patterns without descriptor recommendations, while the determination step receives the full evidence including tiered rankings.
𝒮
param
\mathcal{S}_{\mathrm{param}}
specifies optimal dimensionality controls and additional tunable parameters for all 75 descriptor–object-type combinations, serving as reference configurations for the LLM during Reasoning and as deterministic fallback for Action (see Supplementary for full details).
𝒮
\mathcal{S}
is constructed using classification accuracy as the evaluation metric, a standard protocol for topological descriptors
[
5
]
. We evaluate on 26 TopoBenchmark datasets (§
3.4
) with 6 classifiers drawn from the top of the TabArena leaderboard
[
17
]
(TabPFN
[
26
]
, XGBoost
[
13
]
, CatBoost
[
48
]
, Random Forest
[
8
]
, TabM
[
21
]
, RealMLP
[
27
]
) and 5-fold cross-validation. The construction proceeds in three stages. (A) Grid search identifies optimal parameter and dimensionality configurations for each descriptor on each dataset, yielding
𝒮
param
\mathcal{S}_{\mathrm{param}}
. (B) Using these optimized configurations, we evaluate all 15 descriptors across 26 datasets and 6 classifiers (
26
×
15
×
6
=
2,340
26\times 15\times 6=2{,}340
balanced accuracy values), and derive per-object-type tiered rankings from each descriptor’s best-performing classifier per dataset, averaged across datasets of the same object type, yielding
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
. (C) Five domain experts analyzed the quantitative results and formulated the qualitative components of both
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
and
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
: descriptor properties, parameter reasoning heuristics, reasoning chains, and PH signal definitions, all grounded in TDA principles.
To prevent information leakage and improve generalization, all components of
𝒮
\mathcal{S}
are organized at the object-type level rather than encoding dataset-specific information. The agent never sees dataset identities, class labels, or per-dataset accuracy values at inference time. For example,
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
records the persistence landscape ranks in Tier 1 for
glands/lumens
, not that it attains a specific accuracy on any dataset. While
𝒮
\mathcal{S}
shares similarities with
retrieval-augmented generation (RAG), two key differences
distinguish it: asymmetric information access during
proposal prevents anchoring (§
3.2
),
and Reflection validates outputs against
quality criteria, which standard RAG pipelines lack.
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
: Descriptor Properties
Example: Persistence Image
Best when:
Both
H
0
H_{0}
and
H
1
H_{1}
content
Weakness:
Wastes dimensions on empty
H
1
H_{1}
;
σ
\sigma
tuning critical
Reasoning hint:
“If avg persistence
<
<
0.01, use
σ
\sigma
= 0.5–0.6 to smooth short-lived noisy features.”
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
: Empirical Evidence
Tiered rankings
(
discrete_cells
):
Tier 1:
MinkowskiFn, TemplateFn
Tier 2:
EulerCurve, BettiCurves
Tier 3:
Silhouettes, PersEntropy
Reasoning chain
(
H
0
H_{0}
-dominant):
“Discrete objects create rich
H
0
H_{0}
; signal is in
H
0
H_{0}
birth/death distribution.”
PH signal rule:
“
β
1
/
β
0
>
1.8
∧
H
1
\beta_{1}/\beta_{0}>1.8\wedge H_{1}
cnt
>
500
>500
→
\to
pers_landscape
.
”
𝒮
param
\mathcal{S}_{\mathrm{param}}
: Validated Parameters
Example: Persistence Image
vessel_trees:
res=14, dim=392D
σ
\sigma
=0.05, weight=squared
glands_lumens:
res=26, dim=1352D
σ
\sigma
=0.6, weight=linear
σ
\sigma
varies from 0.05 (thin vessels) to 0.6 (dense glandular tissue)
Skill Set
𝒮
\mathcal{S}
Figure 3
:
Examples of the skill set
𝒮
\mathcal{S}
using persistence image.
𝒮
prop
\mathcal{S}_{\mathrm{prop}}
encodes qualitative descriptor knowledge,
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
provides tiered rankings and reasoning chains, and
𝒮
param
\mathcal{S}_{\mathrm{param}}
specifies validated parameters per object type.
Dual Memory.
The static skill set cannot anticipate every possible scenario: unseen imaging modalities, atypical PH profiles, or edge cases may require runtime adaptation. TopoAgent addresses this through dual memory, inspired by the verbal memory mechanism in Reflexion
[
50
]
. Short-term memory
M
s
M_{s}
records tool invocations within a single run, implemented as the accumulated history within the LLM’s context window, preventing repeated failures on the same image. Long-term memory
M
l
M_{l}
accumulates structured diagnostic entries across runs within a dataset: each entry records the failed descriptor, diagnosed cause, and successful correction, enabling the agent to learn from earlier mistakes on the same dataset.
M
l
M_{l}
is reset between datasets to prevent cross-dataset leakage but accumulates within each dataset, enabling within-dataset adaptation.
Algorithm
1
summarizes the complete TopoAgent workflow, showing how the pipeline (§
3.2
) and components (§
3.3
) interact.
Algorithm 1
TopoAgent
1:
Medical image
I
I
, task prompt, time limit
t
max
t_{\max}
2:
Topological feature vector
𝐟
∈
ℝ
n
d
\mathbf{f}\in\mathbb{R}^{n_{d}}
, reasoning trace
ℛ
\mathcal{R}
3:
𝒮
←
LoadSkillSet
​
(
)
\mathcal{S}\leftarrow\textsc{LoadSkillSet}(\,)
;
𝑇𝑜𝑜𝑙𝑠
←
{
perception tools (6)
,
descriptor tools (15)
}
\mathit{Tools}\leftarrow\{\text{perception tools (6)},\;\text{descriptor tools (15)}\}
4:
M
s
←
[
]
M_{s}\leftarrow[\,]
;
M
l
←
LoadLongTermMemory
​
(
)
M_{l}\leftarrow\textsc{LoadLongTermMemory}(\,)
5:
— Perception —
6:
(
h
⁡
(
I
)
,
v
⁡
(
I
)
)
←
PerceptionTools
​
(
I
)
(h(I),\;v(I))\leftarrow\textsc{PerceptionTools}(I)
;
M
s
←
M
s
∪
{
h
⁡
(
I
)
,
v
⁡
(
I
)
}
M_{s}\leftarrow M_{s}\cup\{h(I),\;v(I)\}
7:
o
←
LLM.Perceive
​
(
I
,
M
s
)
o\leftarrow\textsc{LLM.Perceive}(I,\;M_{s})
⊳
\triangleright
Object type
o
∈
𝒪
o\in\mathcal{O}
via vision-enabled LLM
8:
— Reasoning (Proposal) —
9:
(
d
p
,
θ
p
)
←
LLM.Propose
​
(
h
⁡
(
I
)
,
v
⁡
(
I
)
,
o
,
𝒮
prop
)
(d_{p},\;\theta_{p})\leftarrow\textsc{LLM.Propose}(h(I),\;v(I),\;o,\;\mathcal{S}_{\mathrm{prop}})
⊳
\triangleright
Stripped
𝒮
rank
\mathcal{S}_{\mathrm{rank}}
only
10:
— Reasoning (Determination) + Action + Reflection —
11:
while
Elapsed
​
(
)
<
t
max
\textsc{Elapsed}(\,)<t_{\max}
do
12:
(
d
∗
,
θ
∗
)
←
LLM.Determine
​
(
d
p
,
θ
p
,
𝒮
rank
,
M
l
)
(d^{*},\;\theta^{*})\leftarrow\textsc{LLM.Determine}(d_{p},\;\theta_{p},\;\mathcal{S}_{\mathrm{rank}},\;M_{l})
13:
𝐟
←
ExecuteDescriptorTool
​
(
d
∗
,
θ
∗
)
\mathbf{f}\leftarrow\textsc{ExecuteDescriptorTool}(d^{*},\;\theta^{*})
⊳
\triangleright
Action: run descriptor tool
14:
q
←
LLM.Reflect
​
(
𝐟
,
d
∗
,
𝒮
,
M
l
)
q\leftarrow\textsc{LLM.Reflect}(\mathbf{f},\;d^{*},\;\mathcal{S},\;M_{l})
⊳
\triangleright
Quality diagnosis
15:
if
q
.
𝑝𝑎𝑠𝑠
q.\mathit{pass}
then
return
(
𝐟
,
{
h
⁡
(
I
)
,
v
⁡
(
I
)
,
o
,
d
p
,
d
∗
,
θ
∗
,
q
}
)
(\mathbf{f},\;\{h(I),v(I),o,d_{p},d^{*},\theta^{*},q\})
16:
end
if
17:
M
l
.
𝑢𝑝𝑑𝑎𝑡𝑒
⁡
(
d
∗
,
q
)
M_{l}.\mathit{update}(d^{*},\;q)
⊳
\triangleright
Record diagnosis + correction
18:
end
while
19:
(
𝐟
,
ℛ
)
←
Fallback
​
(
𝒮
rank
,
o
)
(\mathbf{f},\;\mathcal{R})\leftarrow\textsc{Fallback}(\mathcal{S}_{\mathrm{rank}},\;o)
⊳
\triangleright
Top-ranked descriptor for
o
o
20:
return
(
𝐟
,
ℛ
)
(\mathbf{f},\;\mathcal{R})
3.4
TopoBenchmark
TopoBenchmark serves two purposes: it provides an empirical basis from which the skill set
𝒮
\mathcal{S}
is distilled, and it defines a frozen testbed for evaluating the agent’s descriptor determinations. As discussed in §
1
, existing studies lack consistent protocols for cross-descriptor comparison. TopoBenchmark addresses this gap with standardized evaluation across diverse morphologies.
We curate 26 publicly available 2D medical image classification datasets spanning 11 imaging modalities and 11 clinical domains (full details in the Supplementary), organized into five object types that cover the major morphological categories in medical imaging:
cells
(6 datasets),
glands/lumens
(6),
organ shapes
(8),
vessel trees
(3), and
surface lesions
(3). These five categories exhibit distinct topological signatures that motivate adaptive descriptor determination: discrete cells have the most balanced
H
0
H_{0}
/
H
1
H_{1}
ratio (
β
1
/
β
0
≈
1.3
\beta_{1}/\beta_{0}\approx 1.3
) with the lowest total feature count (
∼
{\sim}
2,090 persistence pairs per image), vessel trees show the strongest
H
1
H_{1}
dominance (
β
1
/
β
0
≈
1.9
\beta_{1}/\beta_{0}\approx 1.9
) from branching vasculature, and glands/lumens give the highest total feature count (
∼
{\sim}
4,842) from complex tissue architecture. Organ shapes exhibit the highest within-type variance, from 602 total pairs (OrganAMNIST) to 7,803 (OCTMNIST). No single descriptor performs best across all five types (Table
1
), showing that descriptor determination must be adaptive.
The complete collection contains over 1.2 million images, making full evaluation practically prohibitive. Moreover, the dataset
sizes vary by three orders of magnitude (516 to 327,680), making a uniform sample size inappropriate: too small wastes discriminative
power on large datasets, and too large exceeds the smallest datasets entirely. We address this with convergence analysis: for each dataset, we evaluate the top-3 descriptors at increasing sample sizes
n
∈
{
50,100,200
,
…
,
10
,
000
}
n\in\{50,100,200,\ldots,10{,}000\}
(
n
max
n_{\max}
capped at the dataset size) using 5-fold cross-validation with TabPFN
[
26
]
across 3 seeds, and define
n
∗
n^{*}
as the smallest
n
n
satisfying three criteria simultaneously: balanced accuracy within 1% of the value
at
n
max
n_{\max}
, standard deviation across the seeds below 2%, and descriptor ranking agreement (Spearman
ρ
=
1.0
\rho=1.0
among the
top-3 descriptors). For example, PCam (5,000 histopathology patches) requires the full
n
∗
=
5,000
n^{*}=5{,}000
for accuracy convergence, while IDRiD (516 retinal images) converges at
n
∗
=
500
n^{*}=500
. However, descriptor ranking stabilizes much earlier (
n
=
100
n=100
for PCam, and
n
=
50
n=50
for IDRiD), meaning the relative ordering of the descriptors converges well before the absolute accuracy plateaus. The resulting frozen benchmark contains 113,182 samples with per-dataset sizes of 500–7,500, pre-computed persistent homology caches, and fixed fold indices for reproducible evaluation (the construction pipeline in the Supplementary).
