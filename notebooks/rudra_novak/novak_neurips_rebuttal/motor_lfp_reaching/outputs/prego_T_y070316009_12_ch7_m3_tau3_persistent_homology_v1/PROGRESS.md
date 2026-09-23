# Independent persistent-homology implementation

Authority: the approved plan in the user request, 2026-09-22.

Work is confined to this sibling experiment and new analysis/plot entrypoints.
Existing results, shared code, branch settings, and future defaults are unchanged.
No separate worktree or skill scratch directory: the requested fixed paths and
write boundary take precedence. No commit or push is requested.

## Steps
- Complete: exact input verification; six analytic/input tests and three plot-contract tests pass (observed fail first, then pass).
- Complete: six-trial sequential pilot, all successful (9.41-12.19 seconds per trial).
- Complete: all 198 exact PH calculations successful; no approximations or failures.
- Complete: corrected verifier reproduces all saved tables; resumed all 198 checkpoints without recomputation.
- Complete: 19 separate 600-dpi PNGs; all inspected at reading scale.
- Complete: independent verification of all 129,862 intervals and 304,128 Betti values.
- Complete: plot-only rerun reproduces every PNG byte-for-byte without importing computational models.
- Complete: 252-word report and provenance/quality records; 13 scoped tests pass.

## Interfaces
- Saved clouds -> Ripser: identical 494 x 3 samples, no preprocessing or model input.
- Complete diagrams -> summaries: preserve infinite intervals; no lifetime thresholds.
- Saved fit validity -> radius plots and outlines only: 197 usable fits, 198 PH inputs.
- Numerical tables -> PNGs: plot entrypoint cannot import or call calculation/fitting code.

Initial verification: clean working tree; Ripser 0.6.15 installed; metadata confirm
1 kHz, samples 4000:4500 before GO at 4500. Pilot original IDs: 6, 9, 10, 14, 16, 17.

## Review and verification fixes
- Independent read-only review identified CSV numeric/missing-value parsing,
  missing example-table validation, and a test import-isolation false positive.
- All corrected; 13 scoped tests pass, including source-freeze and resume tests.
- Original analysis source and provenance remain immutable. Verification-only
  revisions are separately archived; an AST check proves numerical functions
  and global calculation settings are unchanged. No trial recomputation occurs.
- The currently active parent loaded the original verifier before its fix. Its
  final CSV check is expected to stop; finish by resuming all saved checkpoints
  with the corrected verifier, not by rerunning Ripser.
- Plot QA so far: label-blind Direction 1 cloud and persistence diagram inspected
  at reading scale, shared limits include all six preselected examples.
- Pilot checkpoint hashes frozen separately to verify exact reuse after completion.
- Full run progress: 157/198 successful, zero failures; original trial 128 retained
  and successfully analyzed despite its earlier geometric nonconvergence.
- Final calculation accounting: 198/198 successful, 197 usable saved radii.
  Original parent stopped at its known CSV-verification bug after writing all
  results; corrected runner exited successfully with 198 reused checkpoints.
  All 25,822 protected previous files match their original hashes.

Final scope: only two new entrypoints and this new sibling experiment have been
added. No existing tracked files changed. No commit or push performed.

Final independent artifact review: no blocking issues; reported values and PNG
metadata/hashes confirmed. Clarified that validation checks saved outputs and
derived summaries rather than implying a second independent PH computation.
