# Pre-GO PSD-selected K-torus decoding, without PCA

## Status and purpose

- Experiment ID: `prego_psd_ktorus_nopca_v1`.
- Status: proposed model specification; no new fits or decoding have run.
- The user approved the workflow: PSD -> candidate K -> validated delay
  embedding -> K-dependent geometry in the full embedding -> decoding.
- This document specifies the previously undefined K-dependent fitter. It
  requires review before implementation; it is not a results report.
- Predecessor: commit `e255c08`, experiment `prego_single_channel`.
- Question: how does this no-PCA, K-dependent representation compare with the
  same single-channel spectral baselines for pre-GO reach-direction decoding?
- No DeepLagField, fixed-six-coordinate assumption, fixed 2-torus assumption,
  pseudopopulations, multichannel decoding, or decoder-informed example choice.

## Location and provenance

The new experiment directory will be the following direct sibling of the old
run, created only when implementation starts:

`/home/nochen/code/NeuralFieldManifold/notebooks/rudra_novak/novak_neurips_rebuttal/motor_lfp_reaching/outputs/prego_psd_ktorus_nopca_v1/`

Use only these immediate subdirectories: `cache`, `tables`, `plots`, `diagnostics`,
and `logs`. Do not nest this experiment inside the old run or a presentation
directory. Put `README.md`, `manifest.json`, `config.json`, and `RESULTS.md` at
the new experiment root. Diagnostic-only previews are not primary results.

The manifest must contain the experiment ID, UTC creation and completion times,
source commit, hashes of all participating source files and input files,
environment/package versions, seeds, exact command, model/feature schema,
parent experiment ID, and a run state. Snapshot the source code used by the run.
Record a digest of the predecessor config, frozen tables, and plots before and
after execution. Read-only access to predecessor trials and folds is allowed;
reuse of its geometry features or torus fits is not.

Creation must fail if the destination already contains a run. An explicit
resume must verify configuration, code, input, selection, and fold hashes
before reusing checkpoints. Configuration changes require a new versioned
sibling directory, never replacement of existing results. Use atomic writes
and a process lock. Do not change global output constants in existing scripts.

## Dataset and controls

Audit all 341 converted macaque reaching LFP recordings independently: 214 M
and 127 T. Reuse the audited raw behavioral GO events, not the erroneous old
converted GO metadata for Monkey M.

- Main analysis: short-delay trials, six directions, final 500 ms before GO.
- M: zero-based samples `[4500:5000]`; T: `[4000:4500]`, at 1000 Hz.
- Existing trial-count eligibility: at least 15 valid trials in every direction.
- Existing eligible cohort: M 115 LFPs / 48 recording days; T 95 / 30 days.
- Reuse exact balanced trial IDs, original behavioral trial IDs, and five
  stratified folds from the predecessor after source-hash verification.
- No pooled channels, new animals, movement-aligned windows, or post-GO data.
- Preserve preprocessing, spectral features, classifier, and random seeds unless
  a separately documented implementation defect makes exact reuse invalid.

This first run is the main short-delay experiment. Long-delay robustness is
deferred, not silently mixed into the results. Report source, eligible, and
geometry-evaluable counts separately. New geometry exclusions must be explained;
do not promise 210 completed recordings before evaluating the new model.

## Training-only selection of K, m, and tau

Selection is separate for every recording and every outer training fold.
Pool training trials only, never concatenate them across temporal boundaries.
Do not use direction labels, held-out trials, decoding scores, or example
appearance to choose geometry parameters.

1. Preserve the existing 2-55 Hz training-trial Welch PSD and fitted broadband
   background procedure. Save raw spectra, residual spectra, candidate peaks,
   prominence, and peak support across 120 training-trial bootstraps.
2. Retain the existing peak-support threshold of 0.50. K is the number of
   supported, resolved peaks, with no silent clipping to four and no forced
   K=1 when no peak is found.
3. Call K a PSD-derived candidate mode count, not a demonstrated topology.
   Log candidate harmonic relationships within the 2 Hz spectral resolution.
   PSD alone cannot establish phase independence. Do not automatically merge
   or count additional harmonics as established independent topological factors.
4. Propose m = 2K + 1 as the declared embedding rule. This is a heuristic,
   not a claim of globally optimal dimension. No fixed-six requirement remains.
5. Evaluate AMI at integer delays of 1-100 samples using training data. Start
   with its first smoothed local minimum. Require at least 300 embedded points
   per 500-sample trial and a full-rank oscillatory delay map with condition
   number at most 100. Both thresholds are explicit design choices.
6. If the first minimum fails these checks, test subsequent AMI local minima
   in ascending delay order. Do not shrink K or m to make a fit possible. If
   none pass, mark the fold unresolved and report it. No default delay or 3D
   fallback. Save every candidate and its rejection reason.

The oscillatory delay map has paired columns cos(2*pi*f_k*j*tau/Fs) and
sin(2*pi*f_k*j*tau/Fs), j=0,...,m-1, for the training-selected frequencies.
Numerical rank and condition diagnostics do not transform or truncate the
point cloud. All m delay coordinates remain inputs to the geometry objective.

Report sensitivity and bootstrap stability descriptively. Do not search for
the K/m/tau setting that maximizes the final held-out decoding score. Record
whether observed windows cover enough phase variation to support interpretation;
a full-rank analytical map is not evidence of adequate empirical phase coverage.

## Proposed K-dependent geometric model

The old 3D elliptical tube fitter cannot be extended simply by changing m.
Use an explicit affine image of a product of K circles in R^m:

    g(theta) = c + sum_k [a_k cos(theta_k) + b_k sin(theta_k)]

Here c, a_k, and b_k are all m-dimensional vectors. K=1 describes an ellipse;
larger K describes the additive oscillatory geometry underlying the paper's
delay-coordinate prediction. This is not the old R1/R2/r tube parameterization.
Its image is an embedded K-torus only under appropriate rank and independent
phase conditions. Degenerate or poorly sampled fits must remain identifiable.

For a trial, use theta_k(t) = 2*pi*f_k*t. Trial-specific phase offsets are
absorbed in the vector coefficients a_k and b_k. Initialize frequencies from
the training-selected peaks and constrain each within +/-2 Hz, the analysis
band, and nonoverlapping midpoint boundaries between adjacent selected peaks.
K and m remain fixed within the outer fold. Fit against every coordinate of
the complete trial delay cloud using bounded variable-projection least squares:
solve linear coefficients at each candidate frequency vector, then optimize
frequencies. Retain every coordinate in the residual objective. Record rank,
conditioning, convergence, objective value, frequency bounds, and phase coverage.

This is a new, locally quasiperiodic geometric model, not a validated replacement
for DeepLagField. Its stationary-within-trial phase model is an assumption to
test. It may fail on transient or drifting dynamics. Its close relationship to
spectral structure must be acknowledged when interpreting added-value results.

## Feature contract

For each mode, retain the upper triangle of the full lag-coordinate shape matrix

    G_k = a_k a_k^T + b_k b_k^T.

This describes that mode's elliptical shape and orientation without depending
on its arbitrary phase origin. Do not rotate the observations into principal
components or retain only selected axes. Add the residual RMSE for each of the
m original delay coordinates and the overall normalized squared residual.

The proposed feature count is K*m*(m+1)/2 + m + 1. Save the exact column names,
units, and dimension with each fold. Do not label these features as the old
15D torus vector. Fixed dimensionality across recordings is unnecessary because
each recording/fold has its own classifier. Within each fold, all training and
test trials must have the identical feature schema and ordered mode identities.

Do not append fitted frequencies directly to the geometry-only feature vector.
Feature scaling, missing-value imputation, and LDA estimation use training trials
only. A failed trial fit is missing geometry, not a zero vector. A fold with no
selected modes, unresolved embedding, or no usable training geometry is not a
successful torus analysis. Exclude such recordings from the matched primary
comparison and disclose them. Do not exclude merely for high error or low F1.

## Decoding and inference

Reuse the existing standardized, shrinkage-LDA evaluation. Evaluate peak power,
peak frequency, power + frequency, all band powers, K-torus geometry,
geometry + power/frequency, and geometry + all band powers on identical trials
and folds. Remove the old PCA feature-control method entirely from this run.

Save held-out predictions before plotting. Report macro-F1, per-direction F1,
and row-normalized confusion matrices. Do not call diagonal recall F1. Give
recording-level mean and SD separately for M and T. Report feature counts and
trial-to-feature ratios because the new geometry representation can be larger.

Preserve the eight planned primary tests: within each monkey, geometry against
power/frequency and against all band powers, plus each fused representation
against its own spectral baseline. Use paired recording-day summaries with
Wilcoxon tests, Holm adjustment across eight tests, and day-cluster bootstrap
intervals for paired improvements. Give LFP, day, and animal counts distinctly.
Use 200 method-specific label permutations with all trainable decoder steps
refit. Freeze geometry selection, which never used direction labels.

The primary methods must use the same geometry-evaluable cohort. Also retain
spectral scores for the original 210-recording eligible cohort so exclusions
are visible. Any comparison with the predecessor uses the intersection of
recording/trial IDs and is labeled exploratory, not a new uncorrected primary
significance claim. This run follows inspection of the predecessor and is not
an independently preregistered confirmation.

## Deliverables

- Root manifest, frozen config, concise results report, and reproduction command.
- Recording/day/trial audit; exclusions with explicit reasons.
- Per-fold PSD peaks, candidate K, m, tau, support and conditioning diagnostics.
- Per-trial fitted parameters, feature vectors, fit quality, and held-out predictions.
- Per-recording and per-animal scores, confusion matrices, null distributions,
  paired differences, bootstrap intervals, and corrected significance tables.
- Separate M/T decoding bars, confusion matrices, and added-value plots, exported
  as PDF, editable SVG, and 600-dpi PNG, with captions and underlying tables.
- Label-blind examples with PSD/AMI diagnostics, observed versus fitted traces,
  and explicitly labeled original-coordinate pair/triple views. These views
  are illustrative slices/projections only, never the inputs to a reduced fit.
- K and embedding-parameter distributions and fit-failure/conditioning summaries.

Plotting must read frozen tables only and must not trigger fitting. Do not
regenerate old movement, poster, presentation, or predecessor outputs.

## Validation gates before the full run

1. Synthetic one-, two-, and three-mode data exercise variable K and full mD
   fitting. Verify feature shape, phase-origin invariance, and reconstruction.
2. A perturbation restricted to the fourth or later delay coordinate must affect
   the full-space objective; this guards against silently using three columns.
3. Explicit cases cover zero peaks, unresolved close peaks, harmonic ambiguity,
   missing AMI minima, rank deficiency, inadequate sample count, and failed fits.
4. Mutating held-out trials or direction labels must not change training-fold
   K, m, tau, peak support, or fitted decoder preprocessing.
5. Input/trial/fold hashes reproduce the predecessor's audited selection exactly.
6. Resume rejects changed code, inputs, configuration, or feature schema.
7. Full-run confusion matrices and F1 reproduce saved held-out predictions.
8. Means, SDs, paired differences, intervals, and p-values reproduce frozen tables.
9. Old-output digests remain unchanged. Inspect new graphics for readability,
   clipping, and correct units. Report unfavorable and unresolved results.

Do not launch the full dataset or claim camera-ready readiness until the model
specification is accepted and the implementation passes these gates.
