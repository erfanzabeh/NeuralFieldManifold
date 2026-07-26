---
name: rudra-iterate-unit-readme
description: Audit and rewrite README files for compact research units in any project. Use when Codex is asked to clean, standardize, review, or iterate unit documentation for analysis, figure, experiment, report, dashboard, model, or debug units, especially when a unit has multiple scripts, outputs, cached artifacts, models, statistics, or plots.
---

# Rudra Iterate Unit README

## Workflow

1. Read the project documentation that governs the unit before editing, such as `AGENTS.md`, `ORGANIZATION.md`, `STYLE.md`, `DECISIONS.md`, data docs, schema docs, or local contribution guides.
2. Inventory the target compact unit: list its scripts/notebooks, generated outputs, cache or intermediate artifacts, inputs, and imported shared helpers.
3. Read each target script and the shared helpers it calls until the calculation, model, statistics, labels, sorting, grouping, and output generation are clear.
4. Rewrite the README from the script behavior, not from folder history. Do not describe parking, wrappers, migrations, file moves, or cache mechanics as the method.
5. For every script that produces a distinct plot/table, create a separate H1 named after that script or output stem. Under each H1, repeat the same H2 sections: `Method`, `Variables`, `Statistics`, `Legends`, `Interpretation`, `Notes`, and `References`.
6. Validate the rewritten README against the scripts: every model, statistic, threshold, label mapping, data signal, axis, color scale, grouping rule, sorting rule, and output should be inferable from the README.

## Section Standard

Use this structure for each script/output:

```markdown
# <script_or_output_stem>

## Method

- Load the exact data source and select the relevant records, entities, samples, observations, trials, sessions, files, or labels.
- State how inputs, labels, entities, samples, time points, positions, or observations are aligned.
- State what measurement, feature, score, summary, or representation is computed from the input signal/data.
- State the model, statistic, or transform used, including cross-validation and null/threshold when applicable.
- State how values are aggregated, sorted, filtered, or summarized.
- State what the script writes as plots or tables.

## Variables

- Data/input: ...
- Sessions/groups: ...
- Labels/targets: ...
- Signals/features/measures: ...
- Parameters/thresholds: ...
- Outputs: ...

## Statistics

- Tests/models: ...
- Null hypothesis: ...
- Alternative hypothesis: ...
- Thresholds/decision rule: ...
- What the statistic means: ...
- Why this statistic is appropriate here: ...

## Legends

- X axis: ...
- Y axis: ...
- Color/value: ...
- Grouping: ...
- Ordering/sorting: ...
- Lines/markers/labels: ...
- Panels: ...

## Interpretation

- ...

## Notes

- ...

## References

- ...
```

## Quality Bar

- Methods must explain the analysis calculation step by step. A reader should know whether the analysis uses SVM, regression, PCA, a statistical test, averaging, sorting, or direct measurement without opening the code.
- Name the exact signal, measurement, target, feature, or derived score instead of using vague labels like activity, value, score, or output.
- `Statistics` must state each statistical test, decoding model, fitted model, null model, shuffle, threshold, hypothesis, or quantitative decision rule. Define null and alternative hypotheses in plain terms, state what the statistic means, and justify why it is appropriate for the unit's comparison. If no statistical test or model is used, say that the output is descriptive.
- For predictive models, state target labels, feature matrix, model type, validation split, metric, reference value, and unit of prediction.
- For dimensionality reduction, state what observations enter the model, whether it is fit separately or globally, how trajectories or embeddings are reconstructed, and what visual encodings represent.
- For statistical tests, state the test, null hypothesis, alternative hypothesis, threshold, statistic meaning, and why the test matches the measurement being compared.
- For grouped units, do not collapse multiple scripts into one generic method. Repeat full sections under each H1 even when sections share content.
- Keep bullets concise but not stingy. Prefer several precise bullets over one vague summary bullet.
