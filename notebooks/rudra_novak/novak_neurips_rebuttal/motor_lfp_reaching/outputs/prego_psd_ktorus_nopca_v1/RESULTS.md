# PSD-selected K-torus decoding, without PCA

## Scope

Single LFP channels, six reach directions, short-delay trials, final 500 ms before GO. K, m=2K+1 and AMI delay are training-fold-specific. No PCA or PINN. All original delay coordinates enter an affine product-of-circles fit; this is not the former 15D tube model.

## Cohort

| Monkey | Eligible LFPs | Geometry-evaluable LFPs | Recording days |
|---|---:|---:|---:|
| M | 115 | 110 | 47 |
| T | 95 | 93 | 30 |

All 341 source LFP recordings were audited (214 M, 127 T). Trial-count eligibility required at least 15 valid short-delay trials per direction before balancing. The matched geometry cohort contains 12,786 trials in M and 12,216 in T; these are recording-trial observations, not distinct behavioral trials pooled across simultaneous channels.

## Main results

Matched geometry-evaluable cohort; mean +/- recording SD. Statistical unit is recording day for paired tests, not independent animal.

| Representation | Monkey M macro-F1 | Monkey T macro-F1 |
|---|---:|---:|
| Peak power | 0.1784 +/- 0.0414 | 0.1694 +/- 0.0334 |
| Peak frequency | 0.1534 +/- 0.0377 | 0.1761 +/- 0.0402 |
| Power + frequency | 0.1866 +/- 0.0550 | 0.2205 +/- 0.0500 |
| All band powers | 0.1858 +/- 0.0455 | 0.1979 +/- 0.0411 |
| K-torus geometry | 0.1801 +/- 0.0420 | 0.1875 +/- 0.0408 |
| Geometry + power/frequency | 0.1899 +/- 0.0472 | 0.2221 +/- 0.0448 |
| Geometry + all bands | 0.1945 +/- 0.0476 | 0.2071 +/- 0.0432 |

Added value: 0 of 4 planned additions show a positive effect significant at Holm p < 0.05. Do not infer improved decoding from positive mean differences alone.

## Planned comparisons

Mean paired differences with recording-day-clustered 95% CIs. Two-sided Wilcoxon on paired day means; Holm correction across eight tests. CIs are unadjusted and target the recording-weighted mean, whereas tests use paired day summaries; a CI excluding zero need not correspond to a significant adjusted p-value.

| Monkey | Comparison | Change in F1 [95% CI] | Holm p |
|---|---|---:|---:|
| M | ktorus - power_frequency | -0.0065 [-0.0174, +0.0042] | 0.9466 |
| M | ktorus - all_bands | -0.0057 [-0.0162, +0.0049] | 0.9296 |
| M | ktorus_power_frequency - power_frequency | +0.0033 [-0.0051, +0.0123] | 0.9466 |
| M | ktorus_all_bands - all_bands | +0.0087 [+0.0005, +0.0170] | 0.5885 |
| T | ktorus - power_frequency | -0.0329 [-0.0451, -0.0185] | 0.003681 |
| T | ktorus - all_bands | -0.0103 [-0.0198, -0.0001] | 0.5885 |
| T | ktorus_power_frequency - power_frequency | +0.0016 [-0.0105, +0.0143] | 0.9466 |
| T | ktorus_all_bands - all_bands | +0.0092 [-0.0005, +0.0196] | 0.9466 |

## Geometry checks

Normalized residual is SSE/TSS in the preprocessed delay cloud, not decoding error. Numerical convergence does not establish a good geometric fit.

- Monkey M: candidate K counts across included folds {1: 206, 2: 270, 3: 65, 4: 9}; m range 3-9; delay median 15 ms; feature count range 10-190. Training trials per feature: median 2.67, range 0.48-15.90. Held-out successful-fit fraction 0.999; median normalized residual 0.811; frequency-bound fraction 0.213.
- Monkey T: candidate K counts across included folds {1: 281, 2: 169, 3: 15}; m range 3-7; delay median 14 ms; feature count range 10-92. Training trials per feature: median 10.00, range 0.84-15.90. Held-out successful-fit fraction 0.999; median normalized residual 0.820; frequency-bound fraction 0.172.
- Unresolved fold reasons: {'unresolved_embedding': 14, 'no_supported_peaks': 4}.

## Interpretation limits

- PSD-derived K is a hypothesis about oscillatory modes, not proof of independent phases or topology. Harmonic ambiguities are logged, not silently resolved.
- The fitter assumes approximately constant modal frequencies within each 500-ms trial. Boundary hits and large residuals limit geometric interpretation.
- Geometry features and the older tube features are different representations. The predecessor comparison is exploratory and cannot isolate PCA as the sole cause of any change.
- Feature counts vary with K and m; no dimension-matched PCA control is used. Shrinkage LDA and permutation references do not eliminate feature-capacity confounds.
- This experiment follows inspection of the predecessor. It is not an independent preregistered confirmation or cross-animal transfer experiment.

## Verification

All stored held-out predictions were reproduced from frozen features. F1 was independently recomputed from class counts; confusion rows, unchanged spectral predictions, 200 null permutations, and predecessor output hashes were checked. See validation.json and the exact source snapshot under logs/source.

## Reproduction

Use the Python executable recorded in manifest.json. The source snapshot and config identify this exact run. The initial primary run used the exact files in logs/source. The documented completion stage used logs/integrity_source to audit every saved fit and append missing fused-method nulls, without changing geometry or held-out predictions. Use report_prego_ktorus.py --render-only to regenerate figures from verified frozen tables/cached fits without training any classifier or geometry model.
