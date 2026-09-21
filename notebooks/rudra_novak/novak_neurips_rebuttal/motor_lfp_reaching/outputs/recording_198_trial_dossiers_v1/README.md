# The four 198-trial recordings

These are four single-channel recordings from two sessions and two monkeys, not four independent sessions.
All have 198 balanced short-delay trials (33 per direction). No recording was selected by F1.
The first listed recording (M channel 1) is the walkthrough example solely by sorted recording ID.

## Reports

- [monkeyM_session-o080506002-22_lfp-1](./monkeyM_session-o080506002-22_lfp-1/Recording_Report.pdf) | [details](./monkeyM_session-o080506002-22_lfp-1/README.md)
- [monkeyM_session-o080506002-22_lfp-2](./monkeyM_session-o080506002-22_lfp-2/Recording_Report.pdf) | [details](./monkeyM_session-o080506002-22_lfp-2/README.md)
- [monkeyT_session-y070316009-12_lfp-7](./monkeyT_session-y070316009-12_lfp-7/Recording_Report.pdf) | [details](./monkeyT_session-y070316009-12_lfp-7/README.md)
- [monkeyT_session-y070316009-12_lfp-8](./monkeyT_session-y070316009-12_lfp-8/Recording_Report.pdf) | [details](./monkeyT_session-y070316009-12_lfp-8/README.md)

`Overview.pdf` / `Overview.png` compares their existing no-PCA decoding results. `recording_summary.csv` contains exact values.
Each folder includes full converted signal arrays, correctly aligned selected raw epochs, audit-processed epochs,
trial identities and folds, all saved spectral/geometric features, every held-out prediction, seven confusion matrices,
per-direction F1, 200 shuffle scores per method, PSD/AMI choices, geometric fits and diagnostics, AR coefficients/poles,
full candidate validation scores, the historical PCA predecessor, and standalone figures.

The no-PCA decoder is the earlier K-dependent shape-matrix model. The OLS audit did not replace it with an AR-derived decoder.
No serialized fitted LDA object was saved by the original pipeline. No new fitting, training, decoding, or significance testing occurs here.
Lag-readout diagnostics are archived only, not used to choose a geometric model.
Group-level day-clustered tests from the full cohort cannot be assigned to these individual recordings.

The complete multi-gigabyte raw animal MATLAB files are not duplicated; source locations/hashes are preserved in the copied run manifests.
The full per-recording converted arrays are included. Earlier movement-aligned experiments remain untouched in the project, not mixed into this package.

`source_manifest.csv` lists exact source paths and hashes. `verification.json` records internal checks.
`source/build_recording_198_dossiers.py` reproduces the package from existing local results only and refuses to overwrite it.
