"""Regenerate the corrected stage figure package from saved results only."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, UNIT

SCRIPTS = (
    "plot_stage_geometry_decoding.py", "plot_stage_f1_heatmap.py",
    "plot_stage_geometry_trajectories.py", "plot_stage_alternative_examples.py",
    "plot_stage_feature_densities.py", "plot_stage_radial_distributions.py",
    "plot_stage_trial_atlas.py", "render_macaque_six_stage_task.py",
    "inspect_stage_geometry_outputs.py",
)


def main():
    assert json.loads((OUTPUT / "validation.json").read_text())["passed"]
    protected = {str(p): sha256(p) for sub in ("inputs", "checkpoints", "models", "null_checkpoints")
                 for p in (OUTPUT / sub).rglob("*") if p.is_file()}
    protected.update({str(OUTPUT / name): sha256(OUTPUT / name) for name in (
        "heldout_predictions.npz", "features.npz", "bootstrap.npz", "permutations.npz",
        "config.json", "source_input_hashes.json")})
    for name in SCRIPTS:
        print(f"Rendering {name}", flush=True)
        subprocess.run([sys.executable, str(UNIT / name)], check=True)
    cartoon = OUTPUT.parent / "macaque_task_six_stages_cue_offset_v2"
    files = sorted(p for p in OUTPUT.rglob("*.png") if "inspection" not in p.parts)
    files.append(cartoon / "macaque_task_six_stages.png")
    inspected = []
    for path in files:
        with Image.open(path) as image:
            assert all(abs(dpi-600) < .1 for dpi in image.info["dpi"]), path
            pixels = np.asarray(image.convert("RGB"))
            ink = np.any(pixels < 245, axis=2)
            assert ink.mean() > .001, path
            assert not np.concatenate([ink[:2].ravel(), ink[-2:].ravel(),
                                       ink[:, :2].ravel(), ink[:, -2:].ravel()]).any(), path
            inspected.append(dict(file=str(path.relative_to(UNIT / "outputs")),
                                  pixels=image.size, dpi=image.info["dpi"], sha256=sha256(path)))
    for p in OUTPUT.rglob("provenance.json"):
        data = json.loads(p.read_text())
        for export in data.get("exports", []):
            assert not export.get("outside_text", []), p
    assert protected == {p: sha256(p) for p in protected}
    write_json(OUTPUT / "figure_validation.json", dict(passed=True, png_count=len(files),
               input_fit_model_hashes_unchanged=True, plotting_called_analysis=False,
               scripts={name: sha256(UNIT / name) for name in SCRIPTS}, png_checks=inspected))
    print(f"Validated {len(files)} publication PNGs; saved fits and decoding unchanged", flush=True)


if __name__ == "__main__":
    main()
