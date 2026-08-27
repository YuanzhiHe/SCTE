# Certifying Emphysema Densitometry from Thick-Slice CT by Residual Arbitration

**Method.** ARC-CT (Arbitrated Reconstruction with Certificates for CT)

## Motivation
Doctors measure emphysema on a chest CT scan by counting how many lung voxels are darker (less dense) than -950 Hounsfield Units, a number called LAA-950, and by reading the 15th percentile of the lung density distribution, called Perc15. Both are just readings off a histogram of densities. Most hospitals, however, store CT reconstructed in thick slices (5 mm or 8 mm) instead of thin ones (1 mm), and a thick slice is an average over the tissue inside it. Averaging narrows the histogram, so both numbers come out wrong.

One research line trains neural networks to turn thick slices back into thin ones; this now demonstrably helps doctors read the scan. Another line leaves the picture alone and statistically corrects the number afterwards. Neither can answer the question that matters for a clinical trial, and the reason is arithmetic: many different reconstructions give exactly the same LAA-950. In particular, a network that simply shifts the whole density histogram down until the count comes out right looks identical, on every image-quality measure and to every reader, to a network that genuinely restored the dark low-density regions that averaging erased.

This is not a hypothetical worry. In our own preliminary run, fine-tuning a restoration model on real thick/thin scan pairs improved its emphysema numbers a lot - and when we compared the weights before and after, the ONLY thing that had really changed was a single number, the output bias, by about -12 HU. Copying that one number into the un-tuned model reproduced the whole improvement, while the restored image's density spread stayed too narrow. The deeper cause is that the network's assumption about how the thick slice was formed is never checked: it assumes a 5 mm averaging window when a fit on 50 real scan pairs says the real window behaves like 6.25 mm. Under a wrong assumption, the mismatch between the network's output and the scan it was given mixes two different errors together, and a shift big enough to change the diagnosis hides inside it.

## Method
### M-OP
*Work out, from the thick scan alone, how it was actually acquired, and say how much of it that explanation misses.*

1. Try every candidate slice width from 2 to 12 mm in 0.25 mm steps: blur the thin estimate by that width, average it down to thick slices, and see which width best reproduces the actual thick scan. Keep the winner, the whole error curve, a robust noise estimate, and the fraction of the scan the winner still fails to explain. 【author decision: fit against the simple up-sampled thick scan, or against the network's own output - report which】

*The through-plane acquisition model: a normalised Gaussian slice-sensitivity profile of full-width-half-maximum w applied along z, followed by an average over r thin slices per thick slice.*
$$ A_w(x)[k] = \frac{1}{r}\sum_{j=0}^{r-1} (k_w * x)[rk + j], \qquad \sum_i k_w[i] = 1 \tag{1} $$


*The operator solver: a grid fit of the effective width from the observed thick series itself, plus the fraction of the observation the fitted physics cannot explain.*
$$ \hat{w} = \arg\min_{w \in \mathcal{W}} \|A_w(\tilde{x}) - y\|_2^2, \qquad \rho = \frac{\|A_{\hat{w}}(\hat{x}) - y\|_2}{\|y - \bar{y}\|_2} \tag{2} $$

   - _Why:_ Everything downstream is judged against this estimate; if you assume the nominal width instead, the assumption error and the network error get mixed together.
2. Swap the fixed slice-width number the network is told for the one just estimated from this scan, standardised the same way for training and inference.
   - _Why:_ The network should reconstruct the volume implied by the acquisition this scan actually had, not by the one written on the box.

### M-CERT
*Turn the leftover disagreement into a small set of named numbers, one of which is readable no matter what acquisition was estimated.*

3. Learn, from training scans only, how much through-plane texture the thick scan implies the thin one should have, and measure how far off that prediction typically is on scans it never saw; if the prediction is no better than guessing the average, do not ship this term at all. 【author decision: what counts as one protocol for calibration purposes】

*The displacement term and the invariance it rests on: over the eroded interior $L_{e}$ of the lung set, the slab average preserves the mean for every width, so a global attenuation offset is identified independently of the width estimate.*
$$ \hat{\delta} = \operatorname*{mean}_{L_e}\big(A_{\hat{w}}(\hat{x}) - y\big), \qquad \operatorname*{mean}_{L_e} A_w(x) = \operatorname*{mean}_{L_e} x \;\; \forall w \tag{3} $$

   - _Why:_ The textbook formula for how much texture averaging removes is off by an order of magnitude on real data, so this yardstick has to be measured rather than derived, and it must carry the honest error bar that comes with that.
4. Mark the lung voxels on the thick grid, shrink that mask inward so no averaged voxel borrows from outside it, then compute three numbers: the plain overall shift, the leftover structural mismatch, and (if calibrated) the texture term - and write them into the same CSV row as the emphysema numbers.

*The single new training term: the displacement penalty, which bills the free lunch the trajectory guardrail otherwise takes.*
$$ \mathcal{L} = \mathcal{L}_{\text{vox}} + w_{dc}\mathcal{L}_{dc} + w_{traj}\mathcal{L}_{traj} + w_{nps}\mathcal{L}_{nps} + \lambda_\delta |\hat{\delta}| \tag{4} $$

   - _Why:_ The overall shift is the part that fakes a good emphysema number, and it is the one part that can be read off no matter which averaging width was estimated.
5. Feed the real thin scan in place of a reconstruction and check each certificate number returns the value it should for a perfect answer; if a number condemns the perfect answer, report it as a limitation of the measuring stick rather than using it to judge models.

*The arbiter: a feasibility verdict on the certificate rather than a score, naming the dominant violated term when it flags.*
$$ \text{verdict} = \begin{cases} \textsf{certified} & |\hat{\delta}| \le \tau_\delta \wedge \rho_{\text{struct}} \le \tau_\rho \wedge |s| \le \tau_s \\ \textsf{flagged} & \text{otherwise} \\ \textsf{abstain} & |L_e| = 0 \end{cases} \tag{5} $$

   - _Why:_ If the yardstick condemns a perfect answer, it is measuring your estimator's bias rather than the model's behaviour - this check caught exactly that twice while designing the method.

### M-GATE
*Penalise the cheap shift during training, and at deployment attach a certify / flag / abstain verdict to the reported emphysema value.*

6. Add the size of the overall shift to the training loss, letting gradients flow through the blur-and-pool operator but not through the width estimate itself, and sweep how strongly it is weighted.
   - _Why:_ It removes the cheapest way for training to make the emphysema number look right without earning it.
7. Turn the certificate into certified / flagged / abstain with stated thresholds, and publish the whole trade-off curve between how many scans get certified and how accurate the certified ones are.
   - _Why:_ A trial needs to know whether this scan's value may be pooled with the others, and that decision should come from the data, not from paperwork.

### M-EVID
*Show the decomposition is what does the work, and that it survives a different slice thickness.*

8. Load each thick/thin pair, convert to real HU units, crop both to the same lung box, slide the thick series by up to +-4 slices to find the alignment that matches best, cut both so the thin one is exactly r times as many slices, and save them as a pair with a one-line audit record. Build the simulated arm from the same thin crops using a known blur width.
   - _Why:_ You need two situations that differ only in whether the averaging process is known, otherwise you can never tell whether a failure came from the physics or from the network.
9. Run all five versions through exactly the same evaluation code, and report the emphysema numbers together with the certificate for every one of them. 【author decision: how many real pairs the bias-only version is allowed to use】
   - _Why:_ One of the five is the un-tuned model with only its output bias corrected - if that alone matches the fine-tuned model, the whole argument of the paper is confirmed.
10. Turn off one component at a time and re-run the same table, so a reader can see exactly which part carries the result.
   - _Why:_ This is what shows the decomposition is doing the work, rather than extra data or a longer training schedule.
11. Do the whole thing again for 8 mm slices with nothing retuned, and report the same table. 【author decision: keep the 8 mm claim to simulated data, since public real 8 mm pairs do not exist】
   - _Why:_ If the width really is estimated rather than assumed, a thicker series should just report a wider width and keep working.
12. Compare the certificate's spread on simulated versus real scans and check whether the unexplained part predicts, case by case, how much worse the emphysema number gets.
   - _Why:_ This shows directly how much of the simulated-to-real performance drop is explained by the part of the scan the estimated physics cannot account for.

