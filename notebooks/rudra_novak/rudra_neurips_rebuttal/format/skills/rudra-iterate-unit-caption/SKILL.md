---
name: rudra-iterate-unit-caption
description: Write, revise, and critique caption.md files for compact research or analysis units. Use when the user asks for figure captions, unit captions, manuscript caption text, caption critique, or caption cleanup, especially when captions should be source-checked against unit READMEs, scripts, plots, tables, statistics, or visual encodings.
---

# Rudra Iterate Unit Caption

Use this skill for compact-unit captions and caption critique. The goal is clear caption text that matches the figure or output, not a second README, a Methods dump, or an inflated claim.

## Core Rules

- Write caption drafts to the owning unit's `caption.md` when working in a repo with compact units.
- Keep methods, variables, statistics, legends, notes, and references in the unit `README.md`; use `caption.md` for caption text and brief caption-specific checks.
- Start with what the figure or output shows, not a grand claim.
- Use plain verbs: shows, uses, marks, indicates, increases, drops, separates, groups.
- Keep the user's structure unless there is a factual issue.
- Include methodological details only when they change how the figure should be interpreted.
- Do not force conclusions unless the user asks for conclusions.
- If including a conclusion, make it concrete and visible in the figure or output.
- Use unit names, panel names, model names, dataset names, metrics, groups, and labels exactly as shown in the unit README, code, plot, table, or user text.
- Be careful with mechanism. Do not claim causation, mechanism, generality, or statistical support unless the unit evidence directly supports it.
- Avoid vague or inflated phrases such as robust, clearly demonstrates, proves, reveals, validates, strong evidence, and mechanistic insight unless the figure specifically supports that wording.

## Source Check

Before writing or revising `caption.md`, inspect the available sources needed for the claim:

- unit `README.md`
- plotting or table script
- produced plot or table, when available
- project `STYLE.md` for visual language
- project `DECISIONS.md` for cross-unit definitions
- user-provided caption notes or manuscript text

Check visual encodings before describing them:

- panel labels and layout
- x axis and y axis
- color/value scale
- line, marker, bar, heatmap, or table encodings
- grouping, ordering, sorting, filtering, or selection rules
- error bars, intervals, significance marks, thresholds, and reference lines

## caption.md Format

Use this structure unless the project specifies another one:

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

If the unit has one panel only, write `- Single panel: ...` under `Panel Notes`.
If no statistics are present, write `- Statistics checked against: none; descriptive output.`
If no uncertainty remains, write `- Remaining uncertainty: none known.`

## Critique Workflow

1. If asked to critique, critique only. Do not rewrite until the user asks.
2. Identify any mismatch between caption text and the source materials.
3. Flag overclaims, missing visual encodings, vague signal names, missing statistics, or unsupported interpretation.
4. Keep critique concrete and tied to the unit files or visible output.

## Revision Workflow

1. Preserve useful user wording and the intended caption shape.
2. Fix factual mismatches against the unit README, script, plot, table, or user notes.
3. Make axis, color, panel, metric, group, and statistic language explicit enough to read the figure.
4. Remove process notes, implementation history, cache mechanics, and internal file-move history.
5. Prefer one polished caption over several alternate versions unless the user asks for options.

## Caption Pattern

Use this order when it fits the figure:

1. One sentence naming the figure subject.
2. Rows, columns, panels, or groups, if needed to read the layout.
3. What each panel encodes.
4. Essential methodological detail.
5. One concrete takeaway, only if the figure supports it or the user asks.

Keep captions direct. A caption can be short if the figure is simple.
