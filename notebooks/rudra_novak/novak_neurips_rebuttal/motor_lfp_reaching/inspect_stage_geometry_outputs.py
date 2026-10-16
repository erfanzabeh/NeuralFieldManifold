"""Audit saved PNGs and plot tables; produce internal visual-review sheets only."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from prego_geometric_fits import sha256, write_json
from stage_geometry_decoding import OUTPUT, COLUMNS


def main():
    root = OUTPUT
    manifest = json.loads((root/"plot_manifest_all.json").read_text())
    assert len(manifest["exports"]) == 53
    assert manifest["plot_source_sha256"] == sha256(Path(__file__).with_name("plot_stage_geometry_decoding.py"))
    points = pd.read_csv(root/"tables/plotted_feature_points.csv")
    features = pd.read_csv(root/"tables/features.csv").set_index("window_index")
    summaries = pd.read_csv(root/"tables/feature_summaries.csv")
    for name in COLUMNS:
        plotted = points[points.feature == name]
        expected = features.loc[features.usable, name]
        assert len(plotted) == len(expected)
        np.testing.assert_array_equal(plotted.window_index.sort_values(), expected.index.sort_values())
        np.testing.assert_allclose(plotted.value, features.loc[plotted.window_index, name])
        for stage, values in plotted.groupby("stage_name"):
            row = summaries[(summaries.stage_name == stage) & (summaries.feature == name)].iloc[0]
            np.testing.assert_allclose([row.q25, row["median"], row.q75], np.quantile(values.value, [.25, .5, .75]))
    findings = []
    for export in manifest["exports"]:
        assert not export["outside_text"], export["file"]
        path = root/"png"/export["file"]
        assert sha256(path) == export["sha256"]
        with Image.open(path) as image:
            assert image.size == tuple(export["pixels"])
            assert abs(image.info["dpi"][0]-600) < .01
            pixels = np.asarray(image.convert("RGB"))
            nonwhite = np.any(pixels < 245, axis=2)
            assert nonwhite.mean() > .001
            edge = np.concatenate([nonwhite[:3].ravel(), nonwhite[-3:].ravel(),
                                   nonwhite[:, :3].ravel(), nonwhite[:, -3:].ravel()])
            assert not edge.any(), f"Content reaches PNG edge: {path.name}"
            findings.append(dict(file=path.name, nonwhite_fraction=float(nonwhite.mean()),
                                 dpi=image.info["dpi"], edge_clear=True))
    inspection = root/"inspection"
    inspection.mkdir(exist_ok=True)
    for page in range(math.ceil(len(findings)/9)):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for ax, entry in zip(axes.flat, findings[page*9:(page+1)*9]):
            with Image.open(root/"png"/entry["file"]) as im:
                im.thumbnail((650, 500))
                ax.imshow(np.asarray(im))
            ax.set_title(entry["file"].removesuffix(".png"), fontsize=8)
        for ax in axes.flat:
            ax.set_axis_off()
        fig.subplots_adjust(left=.01, right=.99, top=.97, bottom=.01, wspace=.04, hspace=.15)
        fig.savefig(inspection/f"contact_{page+1:02d}.png", dpi=140, facecolor="white")
        plt.close(fig)
    write_json(root/"plot_validation.json", dict(passed=True, n_pngs=len(findings),
                all_feature_points_and_quartiles_reproduced=True, png_checks=findings,
                minimum_configured_font_points=8, visual_review_sheets="inspection/contact_*.png"))
    print(json.dumps(dict(passed=True, n_pngs=len(findings), plot_points_reproduced=len(points)), indent=2))


if __name__ == "__main__":
    main()
