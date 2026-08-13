# NeuralFieldManifold Rebuttal Application Presentation Design

## Purpose

Create a self-contained application section for a NeuralFieldManifold lab presentation. The section is designed to follow Kasra's theory presentation and precede Rudra's software/tutorial presentation. It explains the reviewer-driven experiments, the resulting evidence, and the limits of the claims in language appropriate for a neuroscience audience.

## Deliverables

- An editable PowerPoint deck with approximately 15 core slides and 10 backup slides.
- Speaker notes for every slide, including a concise live script and challenge-response guidance where appropriate.
- A one-page rehearsal cheat sheet summarizing the narrative, numerical results, and likely questions.
- A rendered PDF and slide thumbnails for visual verification when the local toolchain permits.

## Narrative

The core story is scientific rather than chronological:

1. The theory predicts a geometric representation of sustained oscillatory structure.
2. The application question is whether recovered geometry contains behaviorally useful information.
3. Mouse EEG tests sleep-state decoding across repeated one-hour sessions.
4. A reviewer-requested macaque motor-cortex analysis tests six-way reach-direction decoding in a distinct task and species.
5. Spectral and geometric features are compared using matched decoding protocols.
6. The results support complementary information in torus geometry while preserving two important qualifications: all-band power can outperform torus features alone, and the EEG no-radius experiment is a feature ablation rather than a full order-recovery test.

## Core Slide Sequence

1. Applications and rebuttal experiments
2. Reviewer questions that motivated the new analyses
3. Common evaluation framework and feature definitions
4. Mouse EEG dataset and session structure
5. Sleep-state decoding consistency across hours
6. EEG torus features versus relevant-band and Average PSD baselines
7. Spectral nuance: comparison with the stronger all-band feature vector
8. EEG geometry/order feature ablation
9. Reviewer-requested macaque reaching experiment
10. Per-trace lag and embedding-dimension selection
11. Six-way reach-direction decoding
12. Paired statistical evidence
13. Replication separately in Monkey M and Monkey T
14. Additive geometry-plus-spectral decoding
15. Conclusions, limitations, and transition to the software tutorial

## Backup Slide Sequence

1. Exact 15 torus features
2. Nonlinear elliptical-torus fitting workflow
3. Spectral feature definitions
4. Cross-validation and balancing protocol
5. Full EEG per-hour heatmap
6. Macaque confusion matrices
7. Full additive-feature comparison and p-values
8. Per-trace tau and dimension distributions
9. Persistent-homology diagnostics
10. Reviewer question-to-evidence matrix

## Numerical Claims

### Mouse EEG

- Twenty-four one-hour sessions were converted; 21 sessions contained enough Wake, NREM, and REM data for balanced decoding.
- Torus features: F1 = 0.670 +/- 0.060 SD.
- Relevant band, defined as delta: F1 = 0.470 +/- 0.050 SD.
- Average PSD: F1 = 0.485 +/- 0.062 SD.
- All-band power: F1 = 0.696 +/- 0.051 SD.
- Torus versus delta: Holm-corrected paired Wilcoxon p = 5.72e-6.
- Torus versus Average PSD: Holm-corrected paired Wilcoxon p = 5.72e-6.
- Full 15D torus versus no-radius 11D geometry: 0.670 +/- 0.060 versus 0.651 +/- 0.072; paired Wilcoxon p = 0.0822.

### Macaque Motor LFP

- Two macaques, 341 unique LFP traces, and 237 traces with all six reach directions.
- Torus features: F1 = 0.181 +/- 0.032 SD.
- Relevant band, defined as beta (13-30 Hz): F1 = 0.157 +/- 0.028 SD.
- Average PSD: F1 = 0.159 +/- 0.030 SD.
- All-band power: F1 = 0.212 +/- 0.049 SD.
- Torus versus beta: Holm-corrected paired Wilcoxon p = 2.32e-15.
- Torus versus Average PSD: Holm-corrected paired Wilcoxon p = 2.97e-13.
- Torus plus beta improves over beta by approximately 0.029 F1.
- Torus plus Average PSD improves over Average PSD by approximately 0.030 F1.
- Torus plus all-band power does not improve over all-band power in the pooled analysis.
- Per-trace tau was chosen from average mutual information with fallbacks; embedding dimension used robust PSD peaks and dimension = 2K+1, capped at 9D. Median tau was 17 ms; selected dimensions ranged from 3D to 9D.

## Required Qualifications

- Describe the separate Monkey M and Monkey T analyses as replication by animal, not cross-animal transfer.
- Describe the 11D EEG comparison as a no-radius feature ablation, not as direct evidence that the model recovered the wrong AR order.
- State that torus features outperform the relevant single band and Average PSD, while the joint all-band power vector is stronger than torus alone in both principal decoding datasets.
- Frame additive results as evidence of complementarity to compact spectral summaries. Do not claim universal superiority over spectral methods.
- Keep F1 as the primary metric and identify error bars as SD across sessions or LFP traces.

## Visual Design

- White background, black and charcoal text, and the paper's dark-red accent color.
- Large plots using existing reviewer-facing PNG files; avoid screenshots of tables when native text can be used.
- One declarative conclusion per slide title.
- Minimal prose on slides, with detailed explanations in speaker notes.
- Use the same color mapping for torus and spectral conditions throughout the deck.
- Include a small footer showing section and slide number, without paper-like decorative clutter.

## Speaker Notes

Each core slide receives:

- A 30-90 second normal script.
- The intended scientific takeaway.
- A transition sentence.
- A short response to the most likely technical objection.

Backup slides receive a concise explanation and the condition under which they should be shown.

## Verification

- Check every numerical statement against the saved CSV tables.
- Confirm that slide titles and notes distinguish F1 from accuracy.
- Render the full deck and inspect slide thumbnails for clipping, unreadable axes, and inconsistent labels.
- Confirm that no slide calls the by-monkey results cross-animal transfer.
- Confirm that the all-band result and order-ablation limitation are stated explicitly.
