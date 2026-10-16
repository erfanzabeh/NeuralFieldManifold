# NeuralFieldManifold Rebuttal Application Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verified 25-slide PowerPoint, speaker-notes document, rendered preview PDF, and rehearsal cheat sheet for the NeuralFieldManifold application and rebuttal experiments.

**Architecture:** A focused Python builder reads the tracked reviewer-facing CSV tables and PNG figures, validates every reported number, and constructs slides from a small scene-graph abstraction. The same scene graph renders native PowerPoint elements and PNG previews, allowing a contact-sheet visual review without relying on LibreOffice. Speaker notes and the rehearsal sheet are generated from the same slide-content records.

**Tech Stack:** Python 3.11 in `/home/nochen/miniconda3/envs/neuralmanifold`, `python-pptx`, Pillow, Matplotlib, pandas, pytest.

## Global Constraints

- Work under `notebooks/rudra_novak/novak_neurips_rebuttal/presentation`.
- Use F1 as the primary decoding metric and label error bars as SD.
- Use the paper's dark-red accent on a white background.
- Describe Monkey M/T results as replication by animal, not cross-animal transfer.
- Describe the 11D EEG comparison as a no-radius feature ablation, not direct order recovery.
- State explicitly that all-band power outperforms torus alone in the principal EEG and macaque comparisons.
- Generate 15 core slides and 10 backup slides.
- Run all Python commands through `/home/nochen/miniconda3/envs/neuralmanifold/bin/python`.

---

### Task 1: Presentation Tooling And Numerical Contract

**Files:**
- Create: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/tests/test_application_presentation.py`
- Create: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/build_application_presentation.py`

**Interfaces:**
- Consumes: reviewer-facing CSV files under the EEG and motor-LFP `tables/` directories.
- Produces: `load_results(root: Path) -> dict[str, Any]` and `validate_claims(results: dict[str, Any]) -> None`.

- [ ] **Step 1: Install the PowerPoint dependency**

Run:

```bash
uv pip install --python /home/nochen/miniconda3/envs/neuralmanifold/bin/python python-pptx
```

- [ ] **Step 2: Write numerical-contract tests**

Create tests asserting the final rounded claims:

```python
assert round(results["eeg"]["torus"]["mean_f1"], 3) == 0.670
assert round(results["eeg"]["all_band"]["mean_f1"], 3) == 0.696
assert round(results["motor"]["torus"]["mean_f1"], 3) == 0.181
assert round(results["motor"]["all_band"]["mean_f1"], 3) == 0.212
assert results["counts"] == {"eeg_valid_sessions": 21, "motor_full_lfps": 237}
```

- [ ] **Step 3: Run the tests and confirm the loader is missing**

Run:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pytest notebooks/rudra_novak/novak_neurips_rebuttal/presentation/tests/test_application_presentation.py -q
```

Expected: failure because `load_results` is not defined.

- [ ] **Step 4: Implement table loading and claim validation**

Read:

```text
tables/eeg_reviewer_feature_summary.csv
tables/eeg_reviewer_relevantband_significance.csv
tables/eeg_torus_order_ablation_summary.csv
tables/eeg_torus_order_ablation_significance.csv
motor_lfp_reaching/tables/nonlinear_refit_direction_movement_summary_pertrace_tau_dim.csv
motor_lfp_reaching/tables/torus_avgpsd_relevantband_significance_pertrace_tau_dim.csv
motor_lfp_reaching/tables/additive_torus_feature_summary_pertrace_tau_dim.csv
motor_lfp_reaching/tables/additive_torus_significance_pertrace_tau_dim.csv
motor_lfp_reaching/tables/lfp_embedding_params_pertrace_tau_dim.csv
```

Map source feature IDs to the presentation names `Relevant band`, `Average PSD`, `All-band power`, and `Torus features`. Raise `ValueError` when required rows or expected sample counts are absent.

- [ ] **Step 5: Run the numerical-contract tests**

Expected: all loader and validation assertions pass.

### Task 2: Dual PowerPoint And Preview Renderer

**Files:**
- Modify: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/build_application_presentation.py`
- Modify: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/tests/test_application_presentation.py`

**Interfaces:**
- Produces: `PresentationBuilder`, `SlideSpec`, `TextElement`, `ImageElement`, `RectElement`, and `LineElement`.
- Produces: `render_pptx(slides: list[SlideSpec], output: Path) -> None` and `render_previews(slides: list[SlideSpec], output_dir: Path) -> list[Path]`.

- [ ] **Step 1: Test scene-graph geometry and slide count**

Assert that a synthetic slide rejects elements outside the 13.333-by-7.5-inch canvas and that `build_slide_specs(results)` returns exactly 25 slides with 15 marked `core` and 10 marked `backup`.

- [ ] **Step 2: Implement the scene graph**

Use immutable dataclasses with inch-based coordinates. Implement aspect-fit and aspect-crop helpers for PNG/JPEG assets. Use Aptos when available in PowerPoint and DejaVu Sans for preview rendering.

- [ ] **Step 3: Implement common slide chrome**

Add a small top-left section label, declarative slide title, red accent line, and bottom-right slide number. Keep all titles at 26-30 pt and body text at 16-20 pt.

- [ ] **Step 4: Implement native PowerPoint rendering**

Use `python-pptx` to create a 16:9 deck. Text, boxes, and rules must remain native PowerPoint objects; analysis plots remain raster images at their source resolution.

- [ ] **Step 5: Implement PNG preview rendering**

Render each `SlideSpec` to 1600-by-900 PNG using Pillow. Use the same coordinates, colors, image fitting, and line wrapping as the PowerPoint renderer.

- [ ] **Step 6: Verify renderer tests**

Expected: geometry, slide count, and smoke-render tests pass.

### Task 3: Core Slide Content And Speaker Notes

**Files:**
- Modify: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/build_application_presentation.py`

**Interfaces:**
- Produces: `build_slide_specs(results: dict[str, Any]) -> list[SlideSpec]`.
- Each `SlideSpec` contains `title`, `section`, `kind`, `elements`, `script`, `takeaway`, `transition`, and `challenge_response`.

- [ ] **Step 1: Build slides 1-3**

Create the section title, reviewer-question triad, and common evaluation pipeline. State balanced five-fold LDA, macro F1, SD, and paired Wilcoxon with Holm correction.

- [ ] **Step 2: Build slides 4-8 for mouse EEG**

Use the EEG summary bar plot, per-hour heatmap, relevant-band comparison, all-band qualification, and no-radius ablation. Include `24 hours`, `21 valid sessions`, and the exact F1/p-value claims from the validated result dictionary.

- [ ] **Step 3: Build slides 9-14 for macaque LFP**

Use `task_sanity.png`, the per-trace embedding figure, the nonlinear six-direction bar plot, the by-monkey heatmap, and the additive-feature plot. Include `341 unique LFPs`, `237 full six-direction LFPs`, median tau `17 ms`, and exact F1/p-value claims.

- [ ] **Step 4: Build slide 15 conclusions**

End with three claims: geometry is behaviorally informative, geometry complements compact spectral summaries, and the strongest multiband baseline remains competitive. Transition directly to Rudra's reproducible software/tutorial section.

- [ ] **Step 5: Write notes for every core slide**

Each slide must contain a 30-90 second script, one-sentence takeaway, transition, and a response to the likely technical objection. The opening must connect to Kasra's theory section.

### Task 4: Backup Slides And Rehearsal Materials

**Files:**
- Modify: `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/build_application_presentation.py`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/NeuralFieldManifold_rebuttal_applications_notes.md`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/NeuralFieldManifold_rebuttal_applications_cheat_sheet.md`

**Interfaces:**
- Produces: `write_notes(slides: list[SlideSpec], output: Path) -> None`.
- Produces: `write_cheat_sheet(results: dict[str, Any], output: Path) -> None`.

- [ ] **Step 1: Build 10 backup slides**

Cover the 15 features, nonlinear torus fit, spectral definitions, CV protocol, EEG heatmap, macaque confusion matrices, additive p-values, tau/dimension distributions, persistent homology, and reviewer evidence matrix.

- [ ] **Step 2: Generate speaker-notes Markdown**

Write one section per slide containing the live script, takeaway, transition, and challenge response. Prefix backup slides with `BACKUP`.

- [ ] **Step 3: Generate the rehearsal cheat sheet**

Limit to one printable page of compact Markdown: opening, six headline numbers, three qualifications, five likely questions with answers, and the closing handoff.

- [ ] **Step 4: Add language tests**

Assert that generated slide text and notes do not contain `cross-animal transfer`, `decoding accuracy` as the primary metric, or a claim that the order-ablation directly measures recovered AR order. Assert that `all-band power` and `feature ablation` both appear.

### Task 5: Build, Render, And Verify The Artifact

**Files:**
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/NeuralFieldManifold_rebuttal_applications.pptx`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/NeuralFieldManifold_rebuttal_applications_preview.pdf`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/previews/slide_*.png`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/previews/contact_sheet.png`
- Create: generated `notebooks/rudra_novak/novak_neurips_rebuttal/presentation/presentation_manifest.json`

**Interfaces:**
- Command: `/home/nochen/miniconda3/envs/neuralmanifold/bin/python build_application_presentation.py`.

- [ ] **Step 1: Run the full builder**

Expected: 25 PNG previews, one contact sheet, one 25-slide PPTX, one preview PDF, notes, cheat sheet, and manifest.

- [ ] **Step 2: Validate the PowerPoint package**

Open the PPTX with `python-pptx`; assert 25 slides and non-empty titles. Inspect the ZIP package for broken media references.

- [ ] **Step 3: Inspect the contact sheet and selected slides**

Check slides 1, 6, 7, 8, 10, 11, 13, 14, 15, and all backup slides for clipping, illegible axes, label overlap, and inconsistent colors.

- [ ] **Step 4: Run the full presentation test suite**

Run:

```bash
/home/nochen/miniconda3/envs/neuralmanifold/bin/python -m pytest notebooks/rudra_novak/novak_neurips_rebuttal/presentation/tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Report deliverables**

Provide clickable paths to the PPTX, notes, cheat sheet, preview PDF, and contact sheet. State any renderer limitation explicitly.
