# <Project Name>

Short project description goes here.

This README is the starter README for a new repo. Keep the initialization section while setting up the repo, then delete it and keep the project README sections below.

## Initialization Section - Delete After Setup

Delete this entire section after the repo has its real name, package, environment, data rules, and first compact units. Everything below this section is the final README skeleton to edit and keep.

### Initialize This Repo

1. Replace `<Project Name>` with the real repo title.
2. Replace the short project description under the title.
3. Rename `project_package/` to the real Python package name.
4. In `pyproject.toml`, replace:
   - `project-template` with the installable package name.
   - `project_package*` with the real package import prefix.
   - the package description with the real project description.
5. In `environment.yml`, replace `project-template` with the conda environment name.
6. In `AGENTS.md` and `ORGANIZATION.md`, replace `<package_name>` with the real package import name.
7. Update `data/README.md` with the real data layout and inventory.
8. Keep `STYLE.md` if this repo will make matplotlib/seaborn figures.
9. Keep `skills/` if this repo should carry the Rudra compact-unit workflow locally.
10. Follow `ref/example_unit/` as the compact-unit reference for cache reuse, `--recompute`, plot output ownership, README detail, and `caption.md` placement.
11. Delete starter-only material listed below.

### Delete After Initialization

Delete these once the project is initialized:

- this whole `Initialization Section`
- `ref/example_unit/`, after the first real compact unit follows the pattern
- `project_package/`, after it has been renamed to the real package

Optional cleanup:

- Delete `server/` if the project has no server/export workflow.
- Delete `STYLE.md` only if the project will not make figures.
- Delete `skills/` only if the repo should not carry local workflow skills.

## Setup

Create the environment from the repo root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate <env_name>
```

Install the package in editable mode if needed:

```bash
python -m pip install -e .
```

## Running Units

Compact units live in `parking/` while they are being explored and in `figs/` after graduation.

Run one unit directly from the repo root:

```bash
python parking/<unit_name>/<unit_script>.py
```

Each unit owns its own `cache/` and `plots/` folders. Existing caches are reused by default. To recompute a unit, delete that unit's relevant cache or run the unit with an explicit recompute option when the script supports one.

## Data

Place organized analysis-ready data at:

```text
data/<real_data_file_or_folder>
```

See `data/README.md` for:

- data file names and locations
- dataset/table layout
- variable meanings
- alignment rules
- inventory of available records

## Project Docs

- `AGENTS.md`: working rules for Codex agents.
- `ORGANIZATION.md`: folder layout, compact-unit pattern, cache rules, and README template.
- `STYLE.md`: figure styling standards.
- `DECISIONS.md`: project-wide scientific or analytical decisions.
- `skills/`: repo-local Rudra workflow skills.
