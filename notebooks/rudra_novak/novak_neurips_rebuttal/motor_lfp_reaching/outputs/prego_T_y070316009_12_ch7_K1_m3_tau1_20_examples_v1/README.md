# T channel 7: fixed K=1, m=3; tau=1-20 ms

Start with `Tau_1_to_20_Observed_Clouds.pdf`: six pages, one selected trial per
page, all 20 delays. The overlay PDF has the same views plus the existing
1-torus planar annular-band fits. `All_120_Enlarged_Comparisons.pdf` contains
one page per trial/delay: observed cloud beside fitted band. Every large
comparison is also exported separately in `individual/` as PDF, SVG, and
300-dpi PNG; overview sheets and source traces are in `figures/`.

## Fixed inputs and interpretation

- Recording: monkeyT_session-y070316009-12_lfp-7, short delay, final 500 ms before GO.
- Six examples: lowest original trial ID in each direction, unchanged from the
  tau=16 report. This is not a sweep of all 198 trials.
- The saved processed epochs are reused exactly. No re-filtering, envelope
  normalization, PCA, AR signal replacement, PINN, or decoding is performed.
- K=1 is a fixed working assumption. m=3 for every plot. At 1000 Hz, one sample
  is one millisecond. All valid lag vectors are retained: 498 points at tau=1,
  468 at tau=16, 460 at tau=20. The earliest endpoint moves with delay; the
  source epoch and final endpoint remain fixed. No trial boundaries are crossed.
- All axes have equal physical scale; limits and camera are fixed across a
  trial's complete sweep and across observed/overlay views. Different example
  trials may have different amplitude limits. There is no per-delay rescaling.
- The existing one_torus_fit objective, bounds, Huber loss, lam=0.1,
  hole_ratio=0.5, three angular penalty harmonics, and 6000-evaluation budget
  are unchanged. Penalty harmonics are not K.
- No best tau is selected. Near-zero delay can produce a nearly collinear
  cloud and small fitting errors without resolving the underlying geometry.
- Common error: squared native 3D distances to the fitted planar band divided
  by total centered cloud sum of squares. This includes out-of-plane error.
  Native coverage and native R-squared ignore out-of-plane displacement.
  These are geometric metrics, not prediction/decoding accuracy or topology tests.
- All failed/nonconverged or boundary-flagged fits remain in checkpoints/tables.

## Reproduction

Run `prego_tau_sweep.py fit` for a NEW experiment directory (existing outputs
are protected), then `prego_tau_sweep.py plot` for plot-only regeneration.
`prego_tau_sweep.py verify` checks arrays, identities, and saved fit metrics.
All commands accept `--output`; fit additionally accepts `--source`.
Frozen inputs, original-coordinate clouds, per-fit parameters/diagnostics,
source snapshots/hashes, and the metric CSV are included here.
