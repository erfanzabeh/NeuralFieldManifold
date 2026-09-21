# PSD-selected K-torus decoding without PCA

This is a separate experiment from `prego_single_channel`. It uses the same
converted reaching recordings, audited raw GO timing, balanced short-delay trial
IDs and five folds, but a different geometry model. No PINN is trained.

## Entry points

- `prego_ktorus.py`: training-only K/m/tau selection, full-coordinate geometric
  fitting, phase-invariant shape descriptors and shrinkage-LDA decoding.
- `run_prego_ktorus.py`: immutable run configuration, source/input hashes,
  checkpoints, spectral reproducibility checks and 200 label permutations.
- `report_prego_ktorus.py`: refits only the LDA from saved feature matrices to
  verify predictions, freezes tables, renders plots and writes the report.
  It never refits geometry or reselects embedding parameters. `--render-only`
  verifies frozen table hashes and does not fit classifiers either.
- `complete_prego_ktorus.py`: independent full-coordinate checkpoint audit and
  completion of the seven-method permutation references, without geometry fits.
- `PREGO_PSD_KTORUS_NOPCA_SPEC.md`: accepted experiment/model definition.
- `PREGO_PSD_KTORUS_EXECUTION.md`: implementation decisions and verification log.

## Run location

The output is a direct sibling, not a descendant, of the predecessor:

`outputs/prego_psd_ktorus_nopca_v1/`

An existing run is refused without `--resume`. Resuming also refuses changed
source code, configuration, raw or converted inputs, predecessor trial caches,
or predecessor frozen tables/figures. Source snapshots and package versions are
saved. New scientific settings require a newly named sibling experiment.

From this directory, using the project's Python environment:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python run_prego_ktorus.py --jobs 12
/home/nochen/miniconda3/envs/neuralmanifold/bin/python complete_prego_ktorus.py --jobs 12
/home/nochen/miniconda3/envs/neuralmanifold/bin/python report_prego_ktorus.py
```

For interrupted work with identical source and configuration:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python run_prego_ktorus.py --resume --jobs 12
```

`--stage fit` and `--stage decode` allow explicit separate stages. `--limit`
creates a partial checkpointed run and never declares the experiment complete.
Plotting and reporting require all 341 source recordings to have completed the
audit/decode stage, including explicit trial-count and geometry exclusions.

For figures only, after reporting has frozen and checked the tables:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python report_prego_ktorus.py --render-only
```

## Exact v1 provenance

The first completed run's exact implementation is under `logs/source/`. A
subsequent independent review prompted checkpoint guardrails (fold-specific
bindings, artifact hashes, original behavioral IDs, and environment checks).
The original runner also produced nulls for five standalone methods only;
the explicit completion stage appends the two fused methods' nulls.

These changes do not alter the scientific core, features, selected embeddings,
or held-out predictions. `logs/integrity_source/` records the completion-stage
code; its audit reconstructs every successful trial's full-coordinate residual
and shape features from the original data and saved coefficients. The current
guarded runner uses configuration schema 2 and is intentionally not accepted
as an identical-code `--resume` of the initial schema-1 run. Keep the initial
snapshot for exact provenance; new runs need a new sibling output directory.

## Interpretation

The model is an affine image of K circular modes in the entire selected lag
space. It is not the former 3D elliptical tube fit and does not use its 15D
feature vector. Numerical linear algebra solves coefficients and checks rank;
it does not project observations into principal components. PSD-derived K is
a candidate mode count, not proof of independent phase coverage or topology.

Compare methods only on the same geometry-evaluable recordings. The tables also
retain spectral results for all 210 trial-count-eligible recordings so selection
effects remain visible. Report failures, larger feature dimensions, fit errors,
frequency-bound hits and nonsignificant comparisons alongside decoding scores.

## Validation

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pytest tests -q
```

Tests cover synthetic K=1/2/3 recovery, every retained coordinate's contribution,
phase-origin invariance, unresolved modes/embeddings, training-only selection,
variable feature counts, absence of PCA, output isolation and stale-cache
rejection. Reporting reproduces predictions and independently checks F1.
