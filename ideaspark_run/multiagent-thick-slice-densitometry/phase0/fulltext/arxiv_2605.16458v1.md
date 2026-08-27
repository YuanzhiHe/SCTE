# Conservative AI for Safety-Sensitive Medical Image Restoration: Residual-Bounded CT-CTA Enhancement for Intracranial Aneurysm-Relevant Signal Recovery

paper_id: arxiv:2605.16458v1
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Generative and restoration models are increasingly used to enhance degraded medical scans, but in 
safety-sensitive applications a restoration model is not free to rewrite its input. Clinically important cues 
are often subtle, and uncontrolled modifications near anatomical boundaries can introduce edits whose 
downstream effect on human interpretation is difficult to bound. This concern motivates a conservative 
AI perspective on medical image restoration: a useful model is one that improves image-level quality while 
constraining the magnitude, location, and stability of its own modifications. From this perspective, 
restoration is closer to controlled correction than to free image generation, and evaluation should 
explicitly probe how much, where, and how reliably the model edits. 
Intracranial CT and CTA imaging is a natural test case for this perspective. CT and CTA are widely used to 
evaluate suspected intracranial aneurysms because of their availability, speed, and broad coverage of 
cerebrovascular anatomy (Thompson et al., 2015; Levinson et al., 2023). Real-world scans, however, are 
often affected by low-dose noise, blur, motion, ring or band artifacts, and reconstruction artifacts, which 
can attenuate the subtle vascular signals that small aneurysms produce on imaging (Din et al., 2023; Yang 
et al., 2021). Because aneurysm-relevant cues frequently sit on or near high-contrast bone–vessel and 
vessel–parenchyma boundaries, an aggressively enhancing model can plausibly improve global metrics 
while smoothing or shifting the very boundaries that are most informative for interpretation (Bizjak & 
Špiclin, 2023; Joo, 2025). This is the regime in which conservative behavior matters most. 
In this work we treat low-quality CT/CTA restoration as a controlled-modification problem rather than a 
maximum-quality reconstruction problem. We train a 2.5D residual-bounded model on synthetically 
degraded CT/CTA inputs, where degradations are designed to imitate plausible CT failure modes rather 
than arbitrary corruptions, and we evaluate the model along several axes that together describe its 
restoration behavior: aneurysm-relevant image recovery, paired comparison against a conventional 
Gaussian baseline, stability under repeated stochastic degradation, the spatial concentration of 
meaningful edits, and external behavior on low-dose CT. We emphasize at the outset that the evaluation 
reported here is computational. It is intended to characterize controlled restoration behavior, not to 
establish clinical diagnostic performance. 
Contributions. This paper makes the following contributions: 
1. A residual-bounded conservative restoration design for low-quality CT/CTA, in which a learned residual 
is added to the original center slice through an edit-control map that limits both the magnitude and 
the spatial extent of modification. 
2. A synthetic degradation pipeline for controlled evaluation, which generates paired degraded–clean 
inputs from relatively clean CT/CTA scans using blur, motion, Poisson–Gaussian noise, ring/band 
artifacts, and edge-streak artifacts. 

3. A multi-level evaluation protocol, combining an aneurysm-relevant image-recovery matrix, paired 
comparison against a Gaussian baseline, Monte Carlo stability testing, anatomical-overlap localization 
of meaningful edits, and external low-dose CT evaluation. 
4. A conservative interpretation of the results, in which we treat modification footprint, stability, and 
anatomical localization as primary evidence about restoration behavior, and treat PSNR as one image-
level proxy among several rather than as a stand-alone endpoint. 
A shorter version of this work has been accepted for presentation at the 2026 3rd International 
Conference on Artificial Intelligence and Future Education (AIFE 2026), Tokyo, Japan. The present 
manuscript reports an expanded computational imaging preprint version of the work.

## Method

This section describes the data sources used in this study, the synthetic degradation pipeline, the residual-
bounded 2.5D restoration model, the training objective, and the multi-level evaluation protocol. The goal 
is a transparent description that supports reproducibility rather than a claim of clinical readiness. 
3.1 Datasets and Evaluation Roles 
Four data sources played complementary roles in model development and evaluation. The development 
pool consisted of CT/CTA scans drawn from a public intracranial aneurysm imaging resource and used 
both for restoration training and for in-distribution image-recovery evaluation. External low-dose CT data 
were used to probe behavior outside the synthetic training setting. Anatomical segmentation masks 
produced by an off-the-shelf segmentation tool were used in a post hoc analysis to ask whether 
meaningful edits concentrated in plausible anatomical regions rather than spreading across the image. 
Synthetically degraded scans, generated from relatively clean inputs, supplied paired degraded–clean 
training data. Table 1 summarizes these data sources and their roles. 
Table 1. Datasets and their roles in model development and evaluation. 
Dataset 
Data Type 
Role in Study
