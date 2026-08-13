# NeuralFieldManifold Rebuttal Applications - Speaker Notes

## Slide 1: From torus geometry to behavior

**Script**

Kasra has established the theoretical link between sustained oscillations and toroidal lag geometry. I will focus on the behavioral tests: first sleep-stage structure and decoding across a full day of mouse EEG, then reach-direction decoding from macaque motor-cortex LFP.

**Takeaway:** The application section asks whether the recovered geometry carries behaviorally useful information.

**Transition:** I will start with the concrete questions raised during review.

**If challenged:** If asked why decoding matters: it is a proof-of-concept behavioral readout, not the primary claim of the paper.

**Delivery cue:** Pause after the title, then explicitly connect to Kasra's theory section.

## Slide 2: Sleep geometry

**Script**

We begin with the geometry itself. Each two-second EEG window is delay-embedded and fit with an elliptical torus, summarized here by the two major radii R1 and R2 and the tube radius r. The distributions shift with behavioral state. NREM is especially distinct in R2 and r, while Wake and REM partially overlap. This is the visual reason to ask whether the geometry can decode sleep stage.

**Takeaway:** The fitted torus changes systematically across Wake, NREM, and REM.

**Transition:** The next question is whether those shifts support class-specific decoding.

**If challenged:** These are feature distributions from the original analysis; they motivate decoding but are not themselves a cross-validated performance result.

**Delivery cue:** Point to R2 and r first; they show the clearest separation.

## Slide 3: Sleep-stage decoding

**Script**

The left matrix is the original proof-of-concept example. The right matrix is the more conservative rebuttal summary: the mean row-normalized out-of-fold confusion matrix across 21 valid one-hour sessions, with the cell-to-cell standard deviation across hours. NREM is the most reliably identified class. Wake and REM are more often confused with one another, and that structure reproduces across the full-day analysis.

**Takeaway:** The class structure in the original example persists across independent recording hours.

**Transition:** We next compare geometry against explicit spectral feature choices.

**If challenged:** These cells are class-wise recalls, not F1 values. The right panel averages normalized matrices across independently decoded hours; it is not one pooled 24-hour classifier.

**Delivery cue:** Read the diagonal, then mention the Wake-REM confusion pattern.

## Slide 4: EEG feature comparison

**Script**

Across 21 valid hours, the 15 torus features reach 0.670 +/- 0.060 SD. They exceed every single-band baseline and the one-scalar Average PSD baseline. Delta is fixed a priori as the physiologically relevant sleep band. The complete all-band vector reaches 0.696 +/- 0.051 SD, slightly above torus alone, so the calibrated claim is that geometry beats compact spectral summaries while the full multiband vector remains a strong baseline.

**Takeaway:** Geometry outperforms each compact spectral baseline; all-band power is strongest overall.

**Transition:** The hour-by-hour view tests whether those averages are driven by only a few sessions.

**If challenged:** Error bars are SD across hours. All-band power is a six-feature vector, not an average of the six band powers.

**Delivery cue:** State the torus result, then immediately give the all-band qualification.

## Slide 5: Across-hour stability

**Script**

Each row is one independently decoded recording hour and each column is one representation. Twenty-one of the 24 hours contain enough Wake, NREM, and REM windows for balanced five-fold decoding. The torus column remains consistently high across the day and is usually stronger than any single frequency band, so the mean effect is not produced by one isolated session.

**Takeaway:** The EEG result repeats across 21 independent one-hour analyses.

**Transition:** We then remove the radius-related information to test which geometric terms matter.

**If challenged:** This is within-hour cross-validation repeated across hours, not train-on-one-hour and test-on-another transfer.

**Delivery cue:** Trace the torus column vertically rather than reading individual cells.

## Slide 6: Radius-feature ablation

**Script**

The complete 15-feature torus vector reaches 0.670 +/- 0.060 SD. Removing the tube radius and three tube-quality terms gives 0.651 +/- 0.072 SD. The paired Wilcoxon p-value is 0.0822, so the numerical decrease is not statistically reliable. This is a targeted feature ablation, not a full refit of an intentionally incorrect autoregressive order.

**Takeaway:** Radius-related terms help numerically, but the ablation difference is not significant.

**Transition:** We now move from spontaneous sleep dynamics to an externally structured motor task.

**If challenged:** Do not call this correct-order versus incorrect-order model accuracy; both bars use cached geometric fits and differ only in retained features.

**Delivery cue:** Say 'feature ablation' before interpreting the p-value.

## Slide 7: Reaching task

**Script**

The reviewer asked whether the framework extends to repeated reaching movements. This public motor-cortex dataset contains two macaques performing six reach directions under short and long delay conditions. The sanity panels show balanced task coverage, GO-aligned LFP spectral structure, and movement onset after the GO cue. The primary analysis decodes six-way reach direction from the movement-aligned LFP epoch across 237 recordings containing all directions.

**Takeaway:** This is a new species, recording modality, and behaviorally instructed task.

**Transition:** Using the same fixed decoder, we compare spectral and torus representations.

**If challenged:** The decoder target is reach direction, not task epoch and not a fabricated eight-condition label; both delay types contribute trials.

**Delivery cue:** Use the three panels only to orient the audience, then move on.

## Slide 8: Reach-direction decoding

**Script**

Across the 237 complete six-direction LFP recordings, torus features reach 0.181 +/- 0.032 SD. Beta power, fixed a priori as the relevant motor band, reaches 0.157 +/- 0.028 SD, and Average PSD reaches 0.159 +/- 0.030 SD. The all-band vector is strongest overall, while torus geometry improves on the compact one-scalar baselines.

**Takeaway:** Torus geometry carries reach-direction information beyond beta power and Average PSD.

**Transition:** The next slide asks whether the same pattern is visible in each animal separately.

**If challenged:** Chance is approximately one-sixth. Report the modest effect size together with the paired consistency across LFPs.

**Delivery cue:** Do not oversell absolute performance; emphasize matched representation comparisons.

## Slide 9: Replicates by animal

**Script**

Separating the data by animal preserves the qualitative result. Torus features exceed beta and Average PSD in both Monkey M and Monkey T. Monkey T has higher overall decodability, particularly for all-band power, but the compact-baseline geometry advantage is not restricted to one animal.

**Takeaway:** Both animals independently show the compact-baseline geometry advantage.

**Transition:** Finally, concatenation tests whether geometry and spectral power are redundant or complementary.

**If challenged:** This is replication by animal, not train-on-one-macaque and test-on-the-other transfer.

**Delivery cue:** Compare rows rather than individual decimal differences.

## Slide 10: Complementary geometry

**Script**

Concatenating torus geometry with beta power or Average PSD significantly improves those compact baselines. Adding geometry to the already richer all-band vector does not produce a significant gain. This is the most precise conclusion: geometry contributes information that compact spectral summaries miss, while a complete multiband representation already captures much of the decodable signal.

**Takeaway:** Geometry complements compact spectral baselines, but not the complete all-band vector.

**Transition:** This closes the application story: interpretable geometry, repeatable behavioral information, and calibrated limits.

**If challenged:** The significance brackets are paired Wilcoxon tests with Holm correction across matched LFP recordings; error bars are SD across LFPs.

**Delivery cue:** End on complementarity, not on a universal claim of superiority.
