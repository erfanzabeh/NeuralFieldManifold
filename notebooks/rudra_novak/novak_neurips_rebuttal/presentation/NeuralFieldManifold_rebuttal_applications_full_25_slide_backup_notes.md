# NeuralFieldManifold Rebuttal Applications - Speaker Notes

## Slide 1: From torus geometry to behavior

**Script**

Kasra has shown the theoretical connection between sustained oscillatory modes and toroidal lag geometry. My part asks the next empirical question: once we recover that geometry, does it carry information about what the brain and animal are doing? The rebuttal gave us two tests: sleep staging across a full day of mouse EEG and six-way reach-direction decoding from macaque motor-cortex LFP.

**Takeaway:** The application section asks whether the recovered geometry carries behaviorally useful information.

**Transition:** I will start with the concrete questions raised during review.

**If challenged:** If asked why decoding matters: it is a proof-of-concept behavioral readout, not the primary claim of the paper.

**Delivery cue:** Pause after the title, then explicitly connect to Kasra's theory section.

## Slide 2: Reviewers asked for evidence beyond a single decoding panel

**Script**

The reviewers did not simply ask for a larger number. They asked whether the result survives a fully specified baseline, whether it is stable across independent recording units, and whether the same idea works in a repeated motor task. Those requests shaped every analysis I will show.

**Takeaway:** The new experiments test rigor and scope, not just performance.

**Transition:** First, here is the common evaluation framework used across both datasets.

**If challenged:** If asked which reviewer prompted what, use the backup evidence matrix; the live story is organized by scientific question.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 3: Both datasets use the same conservative decoding logic

**Script**

For every comparison, we held the decoder fixed and only changed the representation. We balanced the labels, used stratified five-fold LDA, and report macro F1. Error bars are standard deviation across independent sessions for EEG or independent LFP traces for the macaque. Statistical comparisons are paired Wilcoxon tests with Holm correction.

**Takeaway:** Differences in performance reflect representations, not different classifiers.

**Transition:** With that framework fixed, the first test is the full-day mouse EEG recording.

**If challenged:** Macro F1 is appropriate because it weights Wake, NREM, and REM equally after balancing and remains interpretable across six reach directions.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 4: Mouse EEG provides a 24-hour repeated-session test

**Script**

We converted the full 24-hour cortical EEG recording into 24 one-hour sessions. Twenty-one hours contained at least five windows from each sleep class and entered balanced decoding. This gives us repeated temporal replicates rather than one pooled result.

**Takeaway:** The EEG analysis tests repeatability across hours, not just average performance.

**Transition:** The hour-by-hour heatmap shows how stable the representations are.

**If challenged:** The three excluded hours lacked sufficient representation of all three classes; they were marked insufficient rather than forced into cross-validation.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 5: Torus decoding remains stable across recording hours

**Script**

Each row is one valid recording hour, and each column is a feature set. The important pattern is not a perfectly flat torus column; it is that torus decoding remains high across hours and is usually stronger than any single frequency band.

**Takeaway:** The EEG result is distributed across time rather than driven by one session.

**Transition:** Collapsing across hours lets us compare the compact baselines directly.

**If challenged:** These are within-hour cross-validated F1 values summarized across hours; they should not be described as a train-on-one-hour, test-on-another transfer experiment.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 6: EEG geometry outperforms compact spectral summaries

**Script**

Across 21 sessions, torus features reach 0.670 +/- 0.060 SD. Delta, our physiologically relevant sleep band, reaches 0.470 +/- 0.050 SD, and a single scalar Average PSD reaches 0.485 +/- 0.062 SD. Both paired differences survive Holm correction at p = 5.72 x 10^-6.

**Takeaway:** Torus geometry is substantially more informative than delta power or mean broadband PSD alone.

**Transition:** The next slide is the qualification that keeps this claim honest.

**If challenged:** Relevant band is fixed a priori as delta for sleep; it is not selected after seeing decoder performance.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 7: The joint all-band vector is stronger than torus alone

**Script**

When we give the spectral decoder all six band-power values jointly, it reaches 0.696 +/- 0.051 SD, slightly above the torus result. That does not erase the geometry result; it sharpens it. The torus representation beats each single band and Average PSD, but a richer multiband spectral vector remains a strong baseline.

**Takeaway:** The evidence supports complementary information, not universal spectral dominance.

**Transition:** We also tested how much of the torus feature set is actually necessary.

**If challenged:** All-band power is a multifeature vector, not the average of the band powers. Its dimensionality and information content are larger than the one-scalar baselines.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 8: Removing radius-related terms causes a modest, nonsignificant drop

**Script**

The complete 15-feature vector reaches 0.670 +/- 0.060 SD; removing radius and tube-quality terms gives 0.651 +/- 0.072 SD. The paired p-value is 0.0822, so the decrease is not statistically reliable. This experiment is deliberately modest: it is a feature ablation, not direct measurement of wrong AR-order recovery.

**Takeaway:** Radius-related geometry helps numerically, but this shortcut does not establish a significant order effect.

**Transition:** The second application asks whether the framework extends beyond sleep to repeated reaching.

**If challenged:** Do not label these bars correct-order and incorrect-order model accuracy; both are decoders built from cached geometric features.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 9: A reviewer-requested motor task extends the test across species and behavior

**Script**

The reviewer specifically asked whether the approach could be applied to periodic trajectories associated with repeated reaching. We used a public motor-cortex LFP dataset from two macaques. The main analysis uses 237 LFP recordings containing all six reach directions and decodes direction from the movement-aligned epoch.

**Takeaway:** The macaque analysis is a genuinely different task, species, and recording setting.

**Transition:** Before fitting geometry, we made lag parameters trace-specific rather than imposing one global choice.

**If challenged:** The documented task is six reach directions crossed with short and long delay; the primary decoder target here is six-way direction, not an invented eight-condition label.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 10: Lag parameters are estimated independently for each LFP trace

**Script**

For each unique LFP trace, tau comes from the first local minimum of average mutual information, with autocorrelation fallbacks when necessary. The embedding dimension is derived from robust PSD peaks using two coordinates per oscillatory mode plus one, capped at nine dimensions. The median tau is 17 milliseconds, and selected dimensions range from three to nine.

**Takeaway:** The torus fit is not based on a single arbitrary lag or one global dimension.

**Transition:** With those unsupervised parameters fixed, we decode the six reach directions.

**If challenged:** The PSD is used to choose the number of embedding coordinates, not to choose the decoder label or optimize F1.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 11: Torus features decode six reach directions above compact baselines

**Script**

For the 237 full six-direction LFPs, torus features reach 0.181 +/- 0.032 SD. Beta power reaches 0.157 +/- 0.028 SD, and Average PSD reaches 0.159 +/- 0.030 SD. The effect size is modest in absolute terms, as expected for single-channel six-way decoding, but it is consistent across a large number of LFP recordings.

**Takeaway:** Geometry carries reach-direction information beyond beta power and mean PSD.

**Transition:** Matched tests quantify how consistently that advantage appears across LFPs.

**If challenged:** Chance is approximately one-sixth, but macro F1 can sit slightly below nominal chance for noisy multiclass predictions; the paired comparisons are the more informative test.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 12: Small macaque geometry gains are highly consistent

**Script**

The average improvements are about 0.024 F1 over beta and 0.022 over Average PSD. Because these are paired within the same 237 LFPs, we can ask whether the sign and magnitude of the difference are consistent. Both survive Holm correction: p = 2.32 x 10^-15 against beta and p = 2.97 x 10^-13 against Average PSD.

**Takeaway:** The effect is modest but highly repeatable across LFP traces.

**Transition:** The same qualitative pattern also appears separately in each macaque.

**If challenged:** Statistical significance here reflects a large matched sample; pair the p-values with the effect sizes and SDs rather than presenting p-values alone.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 13: The qualitative result replicates separately in both macaques

**Script**

When we summarize Monkey M and Monkey T separately, torus features exceed beta and Average PSD in both animals. Monkey T has higher overall decodability, especially for all-band power, but the compact-baseline geometry advantage is not restricted to one monkey.

**Takeaway:** Both animals independently support the compact-baseline result.

**Transition:** Finally, combining representations asks whether geometry and power are redundant or complementary.

**If challenged:** Call this replication by animal. The decoder is fit and evaluated within each LFP; this is not a leave-one-animal-out transfer analysis.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 14: Geometry complements beta and Average PSD, not all-band power

**Script**

Concatenating torus features with beta increases F1 by about 0.029, and concatenating them with Average PSD increases F1 by about 0.030. Both effects are strongly significant. But adding torus features to the full five-band vector does not improve the pooled result. The all-band vector already captures much of the task information available to this simple linear decoder.

**Takeaway:** Geometry contributes nonredundant information to compact spectral summaries, not an unlimited independent signal.

**Transition:** That leads to the balanced conclusion of the rebuttal experiments.

**If challenged:** Concatenation increases dimensionality. The fair evidence is the paired out-of-fold F1, not the training score; no improvement over all-band power argues against a generic dimension-only benefit.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 15: The rebuttal turns the geometry into a testable application story

**Script**

The application takeaway has three parts. First, the recovered geometry is behaviorally informative in both sleep and reaching. Second, it contributes information beyond compact spectral summaries. Third, the strongest multiband spectral vector remains competitive or stronger, so our claim is complementarity and interpretability rather than universal decoding superiority. The rebuttal made the paper more precise and more useful.

**Takeaway:** The geometry is useful, reproducible, and scientifically interpretable - with clearly stated limits.

**Transition:** Rudra will now show how the full workflow is packaged and how a new user can apply it.

**If challenged:** If challenged on why publish a representation that does not always win decoding: prediction accuracy and geometric fidelity are distinct objectives; the geometry links oscillatory structure to an interpretable state-space object.

**Delivery cue:** Slow down here. This is the slide the audience should remember.

## Slide 16: BACKUP - The torus decoder uses 15 geometric features

**Script**

Use this slide when someone asks exactly what the 15 torus features are. The vector combines scale, fit quality, and pose. It is intentionally compact compared with feeding the entire delay cloud to a flexible model.

**Takeaway:** The torus representation is a compact, interpretable parameter vector.

**Transition:** Return to the relevant result slide.

**If challenged:** Orientation terms are retained in the no-radius ablation because they describe the large fitted ellipse's pose, not local tube thickness.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 17: BACKUP - The reviewer-facing torus fit uses nonlinear surface optimization

**Script**

The final macaque pipeline initializes from PCA, then refines the full elliptical torus by nonlinear least squares. Max nfev equals 1200, meaning the optimizer can evaluate the residual objective up to 1200 times. The result is cached because this is the expensive stage.

**Takeaway:** The reported macaque features come from nonlinear fitted geometry.

**Transition:** Return to the feature comparison.

**If challenged:** PCA is only an initialization and a projection for embeddings above three dimensions; it is not the final torus parameter estimate.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 18: BACKUP - Spectral baselines range from one scalar to a multiband vector

**Script**

This slide clarifies the baseline hierarchy. Average PSD is one scalar; a relevant band is one scalar; all-band power is a vector containing every band power jointly. The richer all-band representation is therefore a substantially stronger baseline.

**Takeaway:** Baseline complexity increases from one scalar to a multiband vector.

**Transition:** Return to the spectral nuance slide.

**If challenged:** Average PSD is log10 of mean Welch PSD after detrending and z-scoring; it is not the average of the decoder scores across bands.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 19: BACKUP - Cross-validation uses matched labels and independent units

**Script**

Each decoding result is cross-validated within its independent reporting unit: hour for EEG and unique LFP for macaque. Features are compared on matched units, then paired statistics are run across those units.

**Takeaway:** The statistics operate on independent session- or LFP-level F1 values.

**Transition:** Return to the relevant p-value slide.

**If challenged:** The existing analysis is not a session-disjoint train/test transfer decoder; describe it as repeated within-unit cross-validation summarized across independent units.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 20: BACKUP - Full EEG heatmap: 21 valid recording hours

**Script**

Use this full-size heatmap when the audience wants the individual hourly values. It exposes both the stability and the real session-to-session variability.

**Takeaway:** The full session matrix is available rather than only a collapsed mean.

**Transition:** Return to the EEG summary.

**If challenged:** Hours 1, 2, and 14 are absent because they lacked enough examples from all three classes.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 21: BACKUP - Macaque confusion matrices show structured direction signals

**Script**

These averaged confusion matrices show that six-way decoding is challenging but not structureless. The torus decoder yields a slightly stronger diagonal than the beta-only decoder across the same LFPs.

**Takeaway:** The macaque F1 difference corresponds to distributed direction information, not one isolated class.

**Transition:** Return to the main macaque comparison.

**If challenged:** A mean confusion matrix averages normalized out-of-fold confusion matrices across LFPs; it is not one pooled classifier.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 22: BACKUP - Additive tests isolate complementary geometric information

**Script**

This table gives the exact primary additive comparisons. Torus features significantly improve beta and Average PSD. They do not improve all-band power in the pooled analysis.

**Takeaway:** Complementarity depends on what the spectral baseline already contains.

**Transition:** Return to the additive bar plot.

**If challenged:** Holm correction is applied within the pooled additive comparison family shown here.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 23: BACKUP - Per-trace lag selection yields heterogeneous dimensions

**Script**

This diagnostic shows why we abandoned a single dataset-wide tau and dimension. The signal structure varies across electrodes and sessions, so each trace receives unsupervised parameters before decoding.

**Takeaway:** Heterogeneity in oscillatory content is modeled rather than averaged away.

**Transition:** Return to the per-trace embedding slide.

**If challenged:** Although embeddings may be higher than three dimensions, the current 15-feature torus fit projects each cloud to its leading three principal coordinates for a comparable elliptical-torus parameterization.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 24: BACKUP - Persistent homology tests the torus hypothesis rather than forcing it

**Script**

This exploratory EEG persistent-homology notebook swept delays, window lengths, and thresholds without hardcoding the answer. In the selected session, the expected two-torus signature was not dominant. That negative result is informative and should not be conflated with the separate fitted-feature decoding analysis.

**Takeaway:** Topological validation and geometric-feature utility are related but distinct empirical questions.

**Transition:** Return to the limitations or conclusions slide.

**If challenged:** Do not use this slide as evidence that every EEG window is a clean topological torus; it demonstrates the diagnostic and its honesty.

**Delivery cue:** Keep the slide conversational and do not read every label.

## Slide 25: BACKUP - The rebuttal converts reviewer concerns into concrete evidence

**Script**

Use this as a navigation slide during questions. It maps each major concern to a result and marks the order-ablation response as partial rather than overselling it.

**Takeaway:** Most application concerns now have a direct table, figure, or diagnostic.

**Transition:** Close by returning to the calibrated conclusion.

**If challenged:** The remaining theoretical noise/window-placement bounds belong in Kasra's theory discussion rather than this application section.

**Delivery cue:** Keep the slide conversational and do not read every label.
