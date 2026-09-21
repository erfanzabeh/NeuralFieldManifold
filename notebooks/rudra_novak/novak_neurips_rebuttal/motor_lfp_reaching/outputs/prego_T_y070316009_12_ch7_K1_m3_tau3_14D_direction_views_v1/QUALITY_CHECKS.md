# Plot-only validation

- Six focused plotting tests passed. These check spatial target order, raw mean/sample-SD summaries, unchanged confusion fractions and instructed-target outlines, complete trial accounting, all nine unscaled orientation components, and absence of analysis imports/calls in the plot entrypoint.
- All seven figure tables reproduce their frozen inputs or arithmetic display summaries. Confusion rows sum to one, each method retains 198 trials, and feature summaries include all 197 usable fits.
- No fitting, decoding, feature extraction, bootstrap, permutation, significance testing, or time-window analysis was run. No new tuning model was fitted. The full analysis test suite was intentionally not run because this request permits plotting only.
- Source CSV hashes and every recorded output hash verified. The renderer checked 420 prior source/plot files and found none changed.
- Every PDF has all extracted text inside its page bounds; minimum font size is 8 pt at the native export size. Each SVG has editable text and vector artwork. Every manuscript PNG is 600 dpi.
- Visually inspected target-space maps, radius profiles, and the orientation heatmap. Fixed color scales and zero-based radius axes are retained; no row rescaling or favorable-feature selection was used. Standalone maps and profiles are available so the wide comparison need not be shrunk to an unreadable size.
- 197 valid feature trials: 33 per direction except direction 3 with 32. The previously nonconverged original trial 128 is omitted from descriptive feature plots without imputation, but remains in the frozen 198-trial predictions through the original training-fold handling.
- Feature ablation and time-resolved decoding were skipped because they require new evaluations.

## Export index

All stems below have PDF, SVG, 600-dpi PNG, CSV, and caption files in `figures/`; smaller PNGs are in `previews/`.

1. `target_map_geometry`
2. `target_map_all_bands`
3. `target_map_comparison`
4. `direction_profile_R1`
5. `direction_profile_R2`
6. `direction_profiles_radii`
7. `orientation_direction_heatmap`

These are exploratory, descriptive views of one recording, not new inferential results.
