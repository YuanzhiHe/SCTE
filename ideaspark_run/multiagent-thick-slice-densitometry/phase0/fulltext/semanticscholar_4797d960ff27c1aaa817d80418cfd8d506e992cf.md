# Impact of slice thickness on CACS calculation with virtual non-contrast in photon-counting CT

paper_id: semanticscholar:4797d960ff27c1aaa817d80418cfd8d506e992cf
tier: T2
source_used: pdf_paper_url_pymupdf
warning: none

## Intro

Coronary artery calcium score (CACS), which is obtained from non-
contrast CT scan, has been demonstrated as an independent prognostic
predictor for all-cause mortality in asymptomatic individuals [1]. The
progression of coronary calcification remained as an independent
predictor for all-cause mortality after adjusting for baseline CACS,
interval time of scan, and clinical risk factors [2]. Nowadays, CACS is
recommended at a class IIa level of the 2019 AHA guidelines for patients
with borderline/intermediate risk and is helpful in guiding clinical
management and treatment [3, 4].
In routine clinical examinations, a separate non-contrast CT scan before
coronary CT angiography (CCTA), was conducted to quantify the CACS,
which may increase the examination time and lead to extra radiation

dose. With the advent of dual-energy and spectral imaging techniques,
several studies [5,6,7] have explored the assessment of the CACS on
virtual non-contrast (VNC) images. These studies using dual-energy
techniques, and have demonstrated high correlations but underestimated
CACS compared to the true non-contrast (TNC) images. With the advent
of photon-counting detector CT (PCD-CT), it has been shown to have
higher spatial resolution and reduced radiation dose. Additionally, PCD-
CT data contain differentiated spectral information, which can be used
for multi-material decomposition [8].
In this context, a novel virtual noniodine (VNI) algorithm has been
developed to create PureCalcium reconstructions on PCD-CT data.
Previous studies have demonstrated that the PureCalcium algorithm
achieves significantly higher accuracy in CACS quantification than VNC
reconstructions [9]. Of note, current virtual non-iodine reconstructions
consistently demonstrate a systematic underestimation of coronary
artery calcium scores compared to true non-contrast reference
standards. While previous investigations have examined the individual
and combined effects of virtual monoenergetic image energy levels and
iterative reconstruction strength, these studies have critically overlooked
the fundamental parameter of slice thickness [10, 11]. This omission
represents a substantial limitation in the existing literature, given that
slice thickness directly governs partial volume effects and significantly
impacts the detectability of small calcifications. Consequently, the
optimal tripartite combination of slice thickness, keV level, and iterative
reconstruction strength for minimizing discrepancies between virtual and
true non-contrast calcium scoring remains undetermined. The systematic

investigation of these three interdependent parameters is therefore
essential for establishing standardized reconstruction protocols that can
enable reliable cardiovascular risk assessment and facilitate the clinical
adoption of virtual non-contrast calcium scoring.In this study, we aimed
to assess the impact of different VMIs, QIR levels, and section thickness
on the accuracy of CACS quantification on the PureCalcium algorithm on
PCD-CT.
Materials and methods
Study patients
A total of 123 patients who underwent separate TNC and contrast-
enhanced photon-counting CCTA due to suspected coronary artery
disease between January 2024 and April 2024 were prospectively
enrolled. Inclusion criteria included subjects older than 18 who
underwent CCTA examinations by standard spatial resolution scanning.
Exclusion criteria were as follows: (1) patients with CACS = 0; (2)
patients who had a history of coronary revascularization (percutaneous
coronary intervention or coronary artery bypass graft). This study was
approved by our ethics committee (YJ-2024-044-1), and written informed
consent was obtained.
Data acquisition
All patients were scanned using a PCD-CT (Naeotom Alpha, software
version VA50, Siemens Healthineers). TNC scans were performed using

the prospective ECG-gated sequential technique, the tube voltage was
set at 120 kVp, and the tube current-time product was set to an IQ level
of 19. CCTA scans were selected as prospective and retrospective
based on the participants’ heart rate and rhythm. The tube voltage was
set at 120 kVp, and the tube current-time product was set to an IQ level
of 64. The contrast materials were applied using bolus tracking with an
enhancement threshold of 100 HU in the descending aorta and a time
delay of 6 s. A total of 45 to 60 mL of an iodinated contrast agent (350
mg iohexol/mL, Omnipaque; GE HealthCare) was injected at a flow rate
of 4.5 mL/sec. Subsequently, 40 mL of saline was injected with the same
rate. If not clinically contraindicated, 0.4 mg of nitroglycerin was
sublingually administered approximately 5 min before the scan [12].
Image reconstruction
TNC images were reconstructed using a section thickness of 3.0 mm
with an increment of 1.5 mm at 70 keV, QIR 2, and a recommended
kernel Qr36, as defined by the vendor’s clinical calcium scoring protocol.
The CCTA images were post-processed using a novel VNI algorithm
(PureCalcium, Siemens Healthineers) with Qr36 kernel. For all of the
patient data, VMIs were reconstructed at 5-keV intervals from 55 to
75 keV with QIR level 1 or 4 for each keV level and section thickness of
3.0 mm with an increment of 1.5 mm or section thickness of 1.5 mm with
an increment of 1.0 mm.
Calcium scoring analysis
The TNC and PureCalcium datasets were evaluated on a semi-

automated workflow (CT CaScoring, syngo.via VB60; Siemens
Healthineers). Calcified lesion attenuation exceeding 130 HU and a
minimum area of 0.5 mm2 was automatically identified as calcium.
Manual adjustments were made as necessary. The calcium scoring
calculation was performed by two radiologists (H.X.Z. and Y.E.Z., with >
10 years of experience in cardiovascular imaging) on anonymized data
sets provided on a syngo.via server. They were blinded to clinical
information and independently assessed all data. Disagreements in
calcium scoring between the two readers were resolved by consensus
reading involving a third senior radiologist (with > 15 years of
experience). A direct comparison was performed between the
PureCalcium algorithm (CACSPC) using prior reconstruction parameters
(55 keV/QIR 1 and 60 keV/QIR 4, 3.0 mm section thickness) [9] and true
non-contrast CT (CACSTNC, 70 keV/QIR 2) to determine the patients with
(CACSTNC> 0) that was not detected by the previous reconstruction
protocol. The smallest mean difference of CACS determined the
combination of optimal reconstruction parameters for calcium detection
compared to the true non-contrast scan.
The burden of calcium plaque in each patient was classified according to
the Coronary Artery Disease Reporting System (P0 = score of 0 [none];
P1 = 1–100 [mild]; P2 = 101–300 [moderate]; P3 = 301–999 [severe]; P4
> = 1000 [extensive]) [10].
Statistical analysis
Continuous data were described as mean ± SD, while categorical data

were expressed as frequencies and proportions. The Kolmogorov-
Smirnov test was used to test the continuous data for normality.
Spearman’s correlation coefficient (r) was used to assess the correlation
between CACSTNC and CACSPC. The differences in calculated CACS
with different reconstruction methods were compared with the referent
standard (CACSTNC) using the Friedman test with Bonferroni correction.
Intraclass correlation coefficient (ICC) and Bland-Altman analysis were
used for agreement assessment. The agreement of plaque burden
classification between CACSTNC and CACSPC was analyzed using
Cohen’s kappa (κ) statistics. Statistical analysis was performed using
SPSS 27 (IBM Corporation, Armonk, NY) and MedCalc 22.0 (MedCalc,
Ostend, Belgium).
Results
Patient characteristics
A total of 123 patients, including 78 males, were enrolled in this study.
The mean age was 69.9 ± 9.5 years, with a range of 45–89 years.
Table 1 summarizes the demographic characteristics of the included
patients in detailed.
Table 1 Characteristics of the enrolled patients
Full size table
Comparison of CACSTNC and CACSPC using different
reconstruction parameters

The median value of CACSTNC was 109.7 (IQR: 36.9–496.9). With the
reconstruction parameters of 3 mm section thickness and an increment
of 1.5 mm, the value of median CACSPC ranged from 83.2 (IQR: 5.5–
399.7; 75 keV, QIR 4) to 132.5 (IQR: 20.2–570.1; 55 keV, QIR 1)
depending on different level of VMI and QIR. For the reconstruction
parameters of 1.5 mm section thickness and an increment of 1 mm, the
median value of CACSPC ranged from 76.9 (IQR: 10.8–364.5; 75 keV,
QIR 4) to 107.4 (IQR: 17.0–474.6; 55 kV, QIR 1). Regardless of whether
the reconstruction section thickness is 3 mm–1.5 mm, CACSPC strongly
correlates with CACSTNC (r = 0.93 and r = 0.96, respectively, P < 0.001)
and shows excellent agreement (ICC between 0.97 and 0.98 for all) for
different keV and QIR levels. No statistical differences were observed in
CACSPC at 3 mm section thickness, 55 keV (QIR4), 60/65 keV (QIR1/4),
and at 1.5 mm section thickness with 55 keV (QIR1/4), 60 keV (QIR1)
compared with CACSTNC (Fig. 1). Detailed results of correlation and
agreement analysis are illustrated in Table 2. The smallest mean bias
was obtained at 1.5 mm section thickness, 55 keV with QIR 1 (CACS:
107.5 [IQR: 18.9–480.4] compared to CACSTNC: 109.7 [IQR: 36.9–496.9]
mean bias, 2.3; LoA, (− 182.7/187.4)) (Fig. 2). Regardless of section
thickness and QIR level, CACSPC values were gradually decreased with
increased keV values (P < 0.001 for all) (Fig. 1).
Fig. 1
[image]
Full size image
Impact of different section thickness, VMI and QIR levels compared with

CACSTNC. Both CACSTNC and CACSpc are illustrated as median. ns
indicates not significant. ***P < 0.001; *P < 0.05; ns > = 0.05. QIR,
quantum iterative reconstruction; VMI, virtual monoenergetic image;
CACS, coronary artery calcium scoring
Table 2 Comparison of CACSTNC and CACSPC with different section
thicknesses, VMI, and QIR
Full size table
Fig. 2
[image]
Full size image
Bland-Altman analysis of CACS compared between TNC and
PureCalicum algori

## Method

impact of different section thickness, level of virtual monoenergetic
images (VMIs), and quantum iterative reconstruction (QIR) on the
accuracy of CACS quantification.
Materials and methods
A total of 123 patients who underwent coronary CT angiography on
PCD-CT with a separate true non-contrast CACS (CACSTNC) scan were
prospectively included. Agatston scores were calculated from the
PureCalcium algorithm (CACSPC) using a section thickness of 3 mm–
1.5 mm, different VMI (55–75 kilo-electron volt (keV)) and QIR (strength
1,4) levels, respectively. CACSTNC at 70 keV and QIR 2 were used as
