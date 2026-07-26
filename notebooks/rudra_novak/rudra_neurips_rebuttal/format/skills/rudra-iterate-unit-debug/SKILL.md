---
name: rudra-iterate-unit-debug
description: Hypothesis-led debugging workflow for uncertain research, analysis, data, model, metric, visualization, or pipeline failures in any project. Use when Codex needs isolated debug artifacts, compact investigation units, explicit hypothesis chains, cheap controlled probes, adversarial result interpretation, expensive-artifact preservation, and deliberate transfer of lessons before changing primary code or results.
---

# Rudra Iterate Unit Debug

Use this skill to investigate uncertain failures without destabilizing the primary system. Favor isolated compact debug units, explicit hypotheses, cheap evidence, and concise conclusions over broad rewrites or speculative fixes.

## Core Rules

- Protect primary code paths, source data, and expensive results unless the user explicitly asks to change them.
- Create an isolated debug area before touching primary logic.
- Treat every idea as a suspicion, not a conclusion, until tested.
- Prefer cheap, controlled probes before expensive full runs.
- Keep a written investigation chain: suspicion -> test -> result -> conclusion -> next suspicion.
- Compare against a known-good control whenever possible.
- Separate real evidence from weak, fake, overfit, or merely suggestive results.
- Transfer lessons deliberately between targets, then test whether they actually transfer.

## Debug Organization

Use a structure like this, adapted to the repo:

```text
debug/
  README.md
  <target>_debug/
    README.md
    scripts/
    runs/
      <run_name>/
        config.json
        sample.csv
        summary.csv
        outputs.csv or sweep.csv
        report.md
```

Use `debug/<target>_debug` or the project's equivalent investigation area for each model, subsystem, dataset, feature, metric, visualization, report, or failure family. Keep reusable scripts under that target's `scripts/` folder unless they are clearly shared across targets.

Use stable, descriptive run names:

```text
<target>_<hypothesis>_<scope>_v1
<target>_<candidate>_validation_v1
<target>_<control>_v1
```

Keep heavy binary artifacts out of commits unless the user explicitly wants them. Prefer committing small configs, CSV summaries, samples, and README conclusions.

## Workflow

### 1. Establish the starting state

Survey the repo, current artifacts, existing results, and known failures. Identify:

- the failing target
- known-good controls
- source data or expensive artifacts that must not be overwritten
- current best result and its metric
- likely confounders such as data alignment, preprocessing, filtering, normalization, sampling, caching, dataset split, metric mismatch, display scaling, prompt wording, tokenization, stale artifacts, schema drift, or encoding issues

Write the starting point in the target README before deep iteration.

### 2. Build the first hypothesis from prior evidence

State one suspicion clearly. Do not bundle many explanations into one test.

Good:

```text
Suspicion: target B may need the same alignment rule that fixed target A.
Test: apply only that alignment rule to target B with existing artifacts and a cheap metric.
Prediction: if the lesson transfers, target B should improve over the current baseline.
```

Bad:

```text
Maybe the data, labels, model, plotting, parameters, and cache are wrong, so try a new pipeline.
```

### 3. Run the cheapest decisive test

Use the smallest experiment that can reject or support the suspicion:

- tiny sample before full sample
- simple invariant or pattern check before judged or expensive evaluation
- narrow parameter band before a wide sweep, unless the relevant range is unknown
- deterministic run before stochastic run when the system allows it
- one variable changed per run

Always include at least one control when practical:

- known-good target under the same harness
- no-intervention baseline
- oracle label, direct instruction, hand-checked sample, or known-good fixture
- old method versus new method
- raw input versus transformed input
- gated versus additive application

### 4. Interpret adversarially

Do not accept metric improvement without inspecting outputs. Look for:

- metric gaming or shortcut behavior
- repeated, malformed, clipped, or impossible outputs
- hidden metadata, schema, or encoding pollution
- improvement concentrated in one subset
- collapse in quality, calibration, validity, or retention
- high score caused by metric artifacts
- overfitting to the tiny sample

Label each result:

- **supported**: improves the intended behavior and passes sanity inspection
- **partial**: improves something real but does not solve the failure
- **rejected**: does not improve the target behavior
- **fake**: improves the cheap metric while outputs are broken or misleading

### 5. Derive the next suspicion from the failure mode

Each failed or partial result should narrow the search.

Examples:

- If a hand-checked control works but the tested method fails, basic capability or data availability may not be the blocker.
- If a larger intervention creates artifacts, the intervention size may not be the real fix.
- If one timing, grouping, preprocessing, or filtering change helps but does not solve the issue, it is only part of the failure.
- If cleaned artifacts help, stored intermediates or preprocessing may be contaminated.
- If late-stage changes create output artifacts, the intervention may be happening too late.

Do not jump to unrelated ideas while the current failure mode has an obvious next test.

### 6. Transfer lessons between targets

When moving from one target to another, start with a transfer table:

| prior lesson | transfer prediction | test | outcome |
| --- | --- | --- | --- |
| A improved after stricter input alignment | B should improve with the same alignment rule | apply alignment only | partial |

If the lesson fails to transfer, preserve it as a negative result and explain the difference. Avoid assuming that a fix is universal because it worked once.

### 7. Validate candidates before recommending primary changes

A candidate is worth recommending only after it survives:

- a broader sample than the discovery probe
- multiple cases, groups, or input families
- output inspection
- comparison to the old method
- a control showing why the new factor matters

If the candidate only works under a cheap metric, say so. Recommend a primary-system experiment, not an immediate permanent change.

## README Format

Each target README should be short and digestible:

```markdown
# <Target> debug

This tree is debug-only. It reads existing artifacts but does not modify primary code, source data, or expensive results.

## Result

Best candidate:

| setting | value |
| --- | --- |
| changed factor | ... |
| control | ... |
| best metric | ... |
| caveat | ... |

## Investigation chain

### 1. Suspicion: ...

Test: ...

Result: ...

Conclusion: ...

### 2. Suspicion: ...

...

## Recommendation

Prioritize ...
Do not prioritize ...
```

Keep run-level `report.md` files factual and mechanical. Keep the target `README.md` as the human narrative.

## Communication

Report progress in concrete terms:

- what context was gathered
- what hypothesis is being tested
- what changed between runs
- what result was observed
- what suspicion follows from that result

Do not overstate early wins. Use phrases like "partial transfer", "cheap metric only", "not ready for primary output", and "candidate for main experiment" when appropriate.

## Stop Conditions

Stop and summarize when:

- a credible candidate has survived validation
- every plausible cheap branch has been rejected
- the next step would require changing primary code, primary outputs, source data, or expensive results
- the next step would require a costly evaluation or data generation the user has not approved

The final summary should include the recommended next main experiment and the rejected alternatives, so future work does not repeat failed branches.
