# Pre-GO single-channel decoding

## Implementation record

User-approved plan: independent panels A-F and S1-S2, six-direction LDA,
short-delay primary analysis, final 500 ms before GO, no pseudopopulations.

- Baseline: existing macaque tests passed (4 tests).
- Existing movement-aligned results and user .gitignore edits must remain untouched.
- Work is on the existing analysis branch unless the user requests a worktree.
- No commits or pushes are part of this request.

## Locked analysis choices

- At least 15 finite, nonconstant pre-GO trials per direction before balancing.
- One shared balanced cohort and five outer folds per recording/analysis.
- Preserve spectral amplitude: detrend raw epochs, do not normalize each trial's
  spectral power by its own amplitude. Fit decoder scaling on training trials.
- Literature peak spectra: 300-sample Hamming segments, 200-sample overlap,
  10000-point FFT at 1000 Hz (0.1-Hz grid, not 0.1-Hz resolving power).
- T peak search: 12-40 Hz. M: 12-25 Hz and 25-40 Hz, retained together.
- All-band powers use the existing five frequency intervals, on raw detrended
  epochs; shared band edges retained for continuity with the earlier pipeline.
- Geometry retains the existing nonlinear elliptical-torus objective (300 points,
  max_nfev=1200), with training-only AMI/PSD selection and a training-only common
  3D projection for embeddings above 3D. Canonicalize radius ordering and axis
  signs so orientation components use consistent coordinates within a fold.
- StandardScaler + shrinkage LDA (lsqr, shrinkage=auto), with training-median
  imputation of failed geometry fits. Report all fit failures and reject a
  recording/analysis if any fold has no usable training geometry. No exclusion
  based on classifier scores. Full 15D and matched-PCA use the same outer folds.
- Nulls permute labels within the fixed outer folds, preserving their class
  counts; rerun the decoder but reuse label-independent fold feature extraction.
  Use 200 permutations per recording/method. Gray references summarize the mean
  across recordings in each permutation; their 95% null interval is not an SD
  error bar or a day-cluster confidence interval.
- S1 balances both delays to the same per-direction count within each recording;
  it is a separate matched analysis, not an unmatched reuse of primary scores.
- Eight two-sided Wilcoxon tests on paired recording-day means; Holm across all
  eight. Cluster-bootstrap mean recording-level differences by day (10000 draws).
- A plot command reads frozen outputs only; it never performs feature fitting.

## Progress

- Core, statistic, and plotting tests written before their implementations.
- Feature extraction, decoding, freezing, statistics, and independent rendering
  completed. All new pre-GO fits and held-out predictions are cached.
- Audited all 341 unique source recordings. The primary cohort contains 115 M
  recordings from 48 days and 95 T recordings from 30 days. The matched-delay
  cohort contains 108 M recordings from 46 days and 94 T recordings from 29 days.
- All 29 non-presentation tests pass. The full suite reports 47 passed,
  10 subtests passed, and three existing, out-of-scope poster failures
  (text inventory/render matching). No presentation files were changed.
- Real-data validation reproduced 597,456 held-out predictions across 614
  recording/analysis combinations and all eight planned comparisons. Every
  standalone export was inspected; PDF/SVG artwork is vector, PNG is 600 dpi,
  and text is at least 8 pt with no clipping. There are 23 artworks including
  two three-radius previews and four supporting panels.

## Observed results

Primary short-delay mean macro-F1 +/- SD across recordings:

| Representation | Monkey M | Monkey T |
|---|---|---|
| Peak power | 0.178 +/- 0.043 | 0.171 +/- 0.035 |
| Peak frequency | 0.154 +/- 0.038 | 0.177 +/- 0.041 |
| Power + frequency | 0.186 +/- 0.055 | 0.221 +/- 0.050 |
| All band powers | 0.185 +/- 0.046 | 0.199 +/- 0.042 |
| Torus features | 0.173 +/- 0.041 | 0.188 +/- 0.034 |

Power + frequency exceeds torus features in Monkey T (Holm-adjusted
p=0.001099 across the eight planned tests). The other three standalone
comparisons are nonsignificant after correction. All four added-value tests
are nonsignificant (Holm-adjusted p=1.0); their day-cluster confidence intervals
include zero. These results do not establish an added decoding benefit of
geometry for this pre-GO, short-delay design.

The primary cohort has 13,338 M and 12,468 T balanced trial observations,
which can share behavioral trials across channels. Exclusions were 99 M and
32 T recordings with fewer than 15 valid short trials in at least one direction;
no recording was excluded for geometry failure or poor decoder performance.
Failed primary geometry vectors were 469/66,690 for M and 274/62,340 for T
(fold-trial vectors, not unique trials); these were retained using training-fold
median imputation. Descriptive radius plots omit failed fits only from display.

## Important source-timing correction

The existing converted files store GO_SAMPLE=4500 for both monkeys. Raw
TrialTimesCorr at code 207 instead reports 5001 for M and 4501 for T (MATLAB
one-based indices). The raw waveform lengths also differ by 500 samples. The
dataset README describes 4501 generically, so it is inconsistent with M's raw
event table. This analysis reads and verifies code 207 for every converted trial,
checks the raw session ID, and uses zero-based M=5000/T=4500. Thus the actual
windows are M [4500:5000] and T [4000:4500]. Nothing in the old conversion or
movement-aligned analyses has been overwritten. Old M analyses require a
separate timing review before comparison with these results.

## Scientific interpretation

This is an adapted single-channel representation comparison with the peak-power
and peak-frequency features of Kilavik et al. (2012), not a reproduction of their
LVQ population decoder, channel preselection, or reported population accuracy.
All methods here use shrinkage LDA and the same trials/folds. No decoding-score
or direction-tuning significance filter is applied. Analyses estimate
within-recording performance, not across-day or cross-animal generalization.

Power features are log10 PSD maxima; frequency features are the corresponding
peak frequencies. All-band powers integrate a 500-sample Hann Welch estimate
over 2-4, 4-8, 8-13, 13-30, and 30-55 Hz before log10 transformation. Geometry
uses the legacy robust trial normalization and 2-55 Hz filter, applied only
inside the cropped epoch. Zero padding does not improve spectral resolution;
the 500-ms window provides limited low-frequency information.

Embedding delay and dimension are selected using training-trial AMI and robust
PSD peaks (120 bootstrap draws). Higher-dimensional embeddings share a training
PCA projection into 3D within a fold. The torus optimizer, sampling budget, and
loss are unchanged. Successful geometry is cached across folds only when their
entire coordinate transform is identical; different training projections never
share fit vectors. Fold 0 supplies the common reference for descriptive panel B.

No inferential claim is made from the descriptive radius examples, per-direction
plots, or supporting dimensionality/delay controls. The eight prespecified
tests use unweighted day-mean paired differences. E's plotted mean gives equal
weight to recordings, and its interval resamples whole days, retaining all
recordings within sampled days. These two summaries have different weightings
when the number of recordings varies by day; both are exported explicitly.
Day differences are rounded to 12 decimal places for the signed-rank test to
avoid numerical subtraction noise creating artificial ranks or nonzero ties.

## Reproduction

Run with `/home/nochen/miniconda3/envs/neuralmanifold/bin/python`. Scripts live in
this directory. Default output is `outputs/prego_single_channel/`.

```bash
python run_prego_single_channel.py --stage fit --jobs 12
python run_prego_single_channel.py --stage decode --jobs 12
python freeze_prego_results.py
python plot_prego_panels.py --panel all
python validate_prego_results.py --jobs 8
```

Alternatively, `--stage all` runs fitting and decoding in one command. The
`--stage decode --wait-for-fits` option can consume a concurrently running fit
stage; do not launch two fit or two decode writers into the same output root.
Completed records/folds are resumed. Configuration changes are rejected rather
than mixed with cached results; use a different `--output` for a changed design.

`plot_prego_panels.py --panel C` (or A-F, S1-S2) reads only checksum-verified
tables. It never loads raw recordings or triggers fitting. Figures are PDF,
SVG with editable text, and 600-dpi PNG at the recorded manuscript dimensions.
Every figure has its own CSV and caption. D/E additionally export their tests.
Individual artwork has no embedded panel letter; filenames identify the panel.

## Sources

- Public data: Kilavik and Riehle (2024),
  https://doi.org/10.12751/g-node.ib7wgy
- Source event definitions, direction order, and duplicate-LFP caveats:
  https://gin.g-node.org/kilavik.b/Macaque_MotorCortex_LFP_Spike_VisuoMotorBehavior/src/master/README.md
- Spectral comparison: Kilavik et al. (2012), Context-Related Frequency
  Modulations of Macaque Motor Cortical LFP Beta Oscillations,
  https://doi.org/10.1093/cercor/bhr299
- Dataset application: Confais et al. (2020),
  https://doi.org/10.1093/texcom/tgaa017
