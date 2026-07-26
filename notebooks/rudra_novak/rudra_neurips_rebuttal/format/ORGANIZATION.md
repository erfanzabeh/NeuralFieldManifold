# Organization

## Root Folders

- `data/`: organized analysis-ready data and its inventory.
- `<package_name>/`: installable project package for shared helpers.
- `parking/`: compact units that are still being explored or iterated.
- `figs/`: graduated compact units that are final figure panels or final outputs.
- `debug/`: investigations, diagnostics, and scratch analyses. One investigation per folder.
- `ref/`: historical notebooks, old pipelines, copied external code, and reference material.
- `server/`: server/export workflows when the project needs them.
- `skills/`: repo-local workflow skills.
- `tmp/`: disposable junk such as archives, zip files, temporary exports, and local leftovers.

## Root Files

- `README.md`: setup, install, data placement, and run entry points.
- `AGENTS.md`: working rules for Codex agents.
- `ORGANIZATION.md`: folder structure, compact-unit pattern, cache rules, and README template.
- `STYLE.md`: figure styling standards.
- `DECISIONS.md`: project-wide decisions that analyses should follow.
- `pyproject.toml`: Python package metadata and pip dependencies.
- `environment.yml`: conda environment for local work.

## Data

`data/` is the canonical source for organized analysis-ready data.

- Keep `data/README.md` current with data layout, variable meanings, alignment rules, and an inventory table.
- Do not move, regenerate, overwrite, or clean data files unless the user explicitly asks.
- Analysis and compact-unit code should read from `data/`, not root-level copies.
- If arrays, tables, labels, records, files, or metadata require an alignment rule, document the exact rule in `data/README.md` and in any unit README that depends on it.

## Python Package

`<package_name>/` is the minimal installable package for reusable project helpers.

- The package is intended for editable installs from this source checkout.
- Do not package `data/`, caches, plots, notebooks, or historical references as package data.
- Shared helpers should be useful across `parking/`, `figs/`, and `debug/`.
- Keep figure-specific calculations, plotting functions, and panel logic inside the compact unit that owns them.
- Do not move unit logic into a shared file just to shorten a unit script.
- Import package helpers from `<package_name>`, not from sibling compact units.
- Do not keep active compatibility shims, migration bridges, or broad holding files for extracted figure code.

## Compact Units

A compact unit is the smallest organized analysis object: one plot, one table, one model, one diagnostic, or one tightly scoped panel with its own code, cache, plots, and README.

```text
parking/descriptive_unit_name/
  descriptive_unit_name.py
  cache/
  plots/
  README.md
  caption.md

figs/fig1a/
  fig1a.py
  cache/
  plots/
  README.md
  caption.md
```

Use descriptive snake_case names in `parking/` while the analysis is still moving. Do not use `p1`, `p2`, `fig1a`, or other figure numbering in parking names. Promote a unit to `figs/figNx` only when the method, plot, legend, and interpretation are stable.

Unit-specific code belongs inside the unit folder. If a unit needs multiple Python files, keep them inside that unit unless the helper is truly shared across multiple units.

Grouping means keeping sibling analyses in one compact unit when they answer the same question with parallel scripts or display variants. A grouped unit still has one `cache/`, one `plots/`, and one `README.md`; each script writes its own named cache and plot outputs inside those folders.

`caption.md` is optional while a unit is still moving, but required once a caption draft exists. Keep caption text in `caption.md`, not in `README.md`.

## Cache Rules

- A compact unit reads existing data and writes only inside its own `cache/` and `plots/`.
- Reuse cache files when present.
- Recompute only when explicitly asked or when the user deletes the relevant cache.
- Do not put new panel caches at the repo root.
- Do not keep a top-level `cache/` folder for active work.
- Cache files are local compute artifacts and are ignored by git unless the user explicitly asks to track one.
- Plot and table outputs in `plots/` are repo artifacts and should be tracked unless the project decides otherwise.
- When regenerating plot outputs, delete that unit's previous plot/table iteration before writing the new one.

## Compact Unit README

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

- What the figure or output shows.
- What comparison matters.

## Notes

- User-specified notes.
- Caveats or pending choices.

## References

- Related units or figures.
- Papers, links, or source artifacts.
```

Each script or output stem that produces a distinct plot/table gets its own H1. Under every H1, repeat the same H2 sections: `Method`, `Variables`, `Statistics`, `Legends`, `Interpretation`, `Notes`, and `References`. A grouped unit with multiple scripts must not collapse them into one generic method.

Keep these READMEs concrete and complete. If `Notes` or `References` has no current content, write `- None yet.`
Never fill `Method` with wrapper history, file moves, cache paths, or script mechanics.
Use `Method` for the conceptual calculation: data selection, alignment, signal extraction, model/statistic/transform, aggregation, sorting/filtering, and written output. Methods must make clear whether the analysis uses SVM, regression, PCA, a direct metric, a statistical test, averaging, sorting, or another operation.
Use `Variables` for exact inputs and choices: data files, labels, windows, thresholds, proportions, model settings, normalization, sorting rules, outputs, and chance/reference values.
Use `Statistics` for every statistical test, decoding model, fitted model, null model, shuffle, threshold, hypothesis, or quantitative decision rule. Name the statistic/model, define the null and alternative hypotheses in plain terms, state the threshold or decision rule, explain what the statistic means, and justify why it matches the measurement and comparison in that unit. If no statistical test or model is used, write `- None; this output is descriptive.` and name any descriptive summaries that appear.
Use `Legends` to make the figure readable without opening the code. Spell out the measured signal behind each color scale.
For predictive models, each relevant H1 must state the target labels, feature matrix, model type, validation split, metric, reference value, and unit of prediction.
For dimensionality reduction, each relevant H1 must state what observations enter the model, whether it is fit separately or globally, how trajectories or embeddings are reconstructed, and what visual encodings represent.
When a unit uses a statistical test, the `Statistics` section must name the test, define the null and alternative hypotheses in plain terms, state the threshold, explain what the statistic means, and justify why that test matches the measurement being compared.

## Compact Unit Captions

Captions live in `caption.md` inside the compact unit that owns the output.

- Use `README.md` for method, variables, statistics, legends, interpretation, notes, and references.
- Use `caption.md` for manuscript-style caption text and brief caption-specific checks.
- Draft or revise captions with the `rudra-iterate-unit-caption` skill when available.
- Source-check caption claims against the unit README, code, produced plots/tables, and project decisions.
- Do not put cache mechanics, implementation history, file moves, or internal process notes in captions.
- Do not claim causation, mechanism, statistical support, or generality unless the unit evidence directly supports it.

Use this structure unless a manuscript target requires another format:

```markdown
# <unit_or_figure_name>

## Caption

<caption text>

## Panel Notes

- A: ...
- B: ...

## Checks

- Visual encodings checked against: ...
- Statistics checked against: ...
- Remaining uncertainty: ...
```

## Decisions

Record only project-wide choices in `DECISIONS.md`. Each section should describe one decision conceptually and include only values or rules that must remain consistent across multiple analyses.

Unit-specific details belong in that compact unit's `README.md`, especially file paths, record subsets, row filters, exclusions, thresholds, statistical tests, statistical-test justification, cache names, and panel-specific mappings.
