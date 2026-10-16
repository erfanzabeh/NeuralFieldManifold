# Corrected six-stage additive comparison

Monkey T, session y070316009-12, channel 7; 198 short-delay trials, six corrected 300-ms stages each.

| Feature set | Dimensions | Held-out macro-F1 | Conditional 95% interval |
|---|---:|---:|---:|
| Relevant band (13-30 Hz) | 1 | 0.291 | 0.267-0.315 |
| Torus + relevant band | 16 | 0.347 | 0.319-0.372 |
| Average PSD | 1 | 0.194 | 0.174-0.213 |
| Torus + average PSD | 16 | 0.311 | 0.284-0.338 |
| Torus features | 15 | 0.293 | 0.268-0.318 |
| All band powers | 5 | 0.400 | 0.375-0.423 |
| Torus + all bands | 20 | 0.401 | 0.372-0.430 |

The [five-bar figure](stage_additive_five_methods.png) follows the requested layout; the [full seven-bar figure](stage_additive_full_comparison.png) also shows all-band powers and geometry added to all bands.
The five-bar figure labels the vertical axis F1; its values are macro-averaged across the six stages.
All methods use identical saved time windows and five trial-grouped folds. Torus features are the unchanged 15D corrected fits; adding beta or average PSD gives 16D, and adding all five bands gives 20D. Spectral features use raw, linearly detrended windows; the torus features came from per-window-normalized, filtered windows. Beta is log10 integrated 13-30 Hz power; average PSD is log10 mean 2-55 Hz density. Five band powers cover delta through low gamma. Hann/Welch uses 300 samples, 150 overlap, and 10,000-point FFT; zero padding does not improve the 300-ms frequency resolution.
The dashed 1/6 line is nominal six-class chance, not a shuffled-label null. Whiskers are 95% percentile intervals from the same 2,000 direction-stratified whole-trial resamples of held-out predictions, conditional on fitted models. This is one channel in one recording; folds or windows are not independent biological replicates. No stars or pooled-animal inference are implied.
Paired additive differences and their conditional intervals are saved in [paired_differences.csv](tables/paired_differences.csv). Beta, mean PSD, and the five band definitions follow the earlier macaque analysis. The geometry decoder reproduces the prior held-out predictions exactly. No geometric refit or Ripser run occurred. The beta and average-PSD features are not matched in dimensionality to the 15D geometry, so standalone scores are a representation comparison, not a controlled per-dimension efficiency test. The geometry-plus-all-bands check was added after the initial five-bar scores were inspected; its near-zero difference is descriptive, not a pre-registered hypothesis test. Low-frequency band estimates are limited by 300-ms windows.
