# Git storage notes

Analysis code, tests, reports, figure artwork, captions, and result tables are tracked. Slide/poster materials, raw NumPy/MATLAB data caches, and process locks remain local under the repository's ignore rules. This repository is not a complete raw-data backup.

## Oversized OLS result table

`outputs/prego_ols_audit_v1/tables/inner_validation_scores.csv` exceeds GitHub's single-file size limit. Its exact bytes are tracked in the lossless archive `inner_validation_scores.csv.gz` beside it. The existing uncompressed local file was not changed or deleted.

After cloning, restore the original filename from this directory when needed:

```bash
gzip -dk outputs/prego_ols_audit_v1/tables/inner_validation_scores.csv.gz
```

The uncompressed SHA-256, also verified against the archive contents, is:

```text
2bc251662e7e269bf967dc407f1841355254e5e339aa3c1eb1fa53482603a054
```

The frozen experiment manifests continue to describe the original uncompressed table. No scientific values or analysis code were altered for compression.

## Local-only bundle

`outputs/recording_198_trial_dossiers_v1.zip` is a redundant convenience bundle containing raw-data caches otherwise excluded by `.gitignore`; it remains local. The dossier folder's nonignored reports, source snapshots, and tables are tracked separately.

Plot-only checks were run before this checkpoint. No experimental fitting, decoding, or statistics were rerun as part of committing and pushing.
