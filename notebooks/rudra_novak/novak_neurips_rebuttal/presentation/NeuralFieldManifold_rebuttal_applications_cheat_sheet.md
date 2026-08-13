# NeuralFieldManifold Application Section - Rehearsal Sheet

## Opening

"Kasra established why sustained oscillatory modes generate toroidal lag geometry. I will ask whether that recovered geometry contains behaviorally useful information."

## Six Numbers To Know

| Result | F1 |
|---|---:|
| EEG torus | 0.670 +/- 0.060 SD |
| EEG delta | 0.470 +/- 0.050 SD |
| EEG Average PSD | 0.485 +/- 0.062 SD |
| Macaque torus | 0.181 +/- 0.032 SD |
| Macaque beta | 0.157 +/- 0.028 SD |
| Macaque Average PSD | 0.159 +/- 0.030 SD |

EEG paired tests: torus vs delta and torus vs Average PSD, both p = 5.72 x 10^-6.  
Macaque paired tests: torus vs beta p = 2.32 x 10^-15; torus vs Average PSD p = 2.97 x 10^-13.

## Three Qualifications

1. All-band power is a multifeature vector and is stronger than torus alone in both principal datasets.
2. The Monkey M/T analysis is replication by animal, not train-on-one-animal/test-on-the-other transfer.
3. The EEG 15D-versus-11D comparison is a no-radius feature ablation, not a full recovered-order experiment.

## Likely Questions

**Why F1?** Macro F1 weights all sleep stages or reach directions equally; labels are balanced before five-fold LDA.

**What are the error bars?** Standard deviation across 21 EEG sessions or 237 unique full-six-direction LFP recordings.

**What is Average PSD?** One scalar: log10 of Welch PSD averaged across the analysis frequency range after detrending and z-scoring.

**How were macaque tau and dimension chosen?** Tau from average mutual information; dimension from robust 1/f-corrected PSD peaks using 2K+1, capped at 9D. Median tau = 17 ms.

**What is the strongest defensible claim?** Torus geometry carries task information beyond compact spectral summaries and adds to beta or Average PSD, but does not universally outperform a complete multiband representation.

## Closing

"The rebuttal turns the theory into a testable application story: useful geometry, reproducible effects, and calibrated claims. Rudra will now show how the workflow is packaged for new datasets."
