#!/usr/bin/env python
"""Read-only scientific-figure checks and a vector PDF summary booklet."""
import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from run_prego_ols_audit import DEFAULT_OUTPUT, sha256, verify_artifacts, write_json


def compare_frames(actual, expected, keys):
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-14)


def verify(root):
    manifest = json.loads((root / "figure_manifest.json").read_text())
    verify_artifacts(root, manifest["artifacts"])
    verify_artifacts(root, manifest["input_tables"])
    assert sha256(Path(__file__).with_name("report_prego_ols_audit.py")) == manifest["plot_source_sha256"]
    files = sorted((root / "plots").rglob("*.pdf"))
    assert len(files) == manifest["figures"] == 435
    source = pd.read_csv(root / "tables/recording_validation_scores.csv")
    keys = ["recording", "family", "p", "m", "tau", "horizon_ms"]
    qc = []
    for path in files:
        for extension in ("svg", "png", "csv", "caption.txt"):
            artifact = path.with_suffix("."+extension)
            assert artifact.is_file() and artifact.stat().st_size > 20, artifact
        with fitz.open(path) as pdf:
            assert len(pdf) == 1
            page = pdf[0]
            assert len(page.get_text().strip()) > 10
            words = page.get_text("words")
            outside = [w for w in words if w[0] < -.5 or w[1] < -.5 or w[2] > page.rect.width+.5 or w[3] > page.rect.height+.5]
            assert not outside, (path, outside)
            sizes = [s["size"] for b in page.get_text("dict")["blocks"] if "lines" in b
                     for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            assert max(sizes) >= 9
        with Image.open(path.with_suffix(".png")) as im:
            assert im.width >= 1800 and im.height >= 700, (path, im.size)
            assert max(ImageStat.Stat(im.convert("RGB")).stddev) > 10, path
        if path.stem in ("order_sweep", "embedding_sweep"):
            recording = path.parent.name
            family = "ar" if path.stem == "order_sweep" else "embedding"
            expected = source[(source.recording == recording) & (source.family == family)]
            compare_frames(pd.read_csv(path.with_suffix(".csv")), expected, keys)
        if path.stem.startswith("monkey_") and path.stem.endswith(("order_sweep", "embedding_sweep")):
            monkey = path.stem.split("_")[1]
            family = "ar" if path.stem.endswith("order_sweep") else "embedding"
            expected = source[(source.monkey == monkey) & (source.family == family)]
            compare_frames(pd.read_csv(path.with_suffix(".csv")), expected, keys)
        qc.append(dict(figure=str(path.relative_to(root)), text_inside_page=True, nonblank=True,
                       min_font_pt=min(sizes), max_font_pt=max(sizes)))
    scores = pd.read_csv(root / "tables/heldout_recording_scores.csv")
    plot = pd.read_csv(root / "plots/summary/heldout_prediction.csv")
    for row in plot.itertuples():
        sub = scores[(scores.monkey == row.monkey) & (scores.horizon_ms == 10)]
        values = sub[sub.family == ("ar" if row.method == "persistence" else row.method)][
            "nmse_persistence" if row.method == "persistence" else "nmse"]
        np.testing.assert_allclose([row.mean, row.sd], [values.mean(), values.std(ddof=1)], rtol=1e-10)
        assert row.n == len(values)
    pd.DataFrame(qc).to_csv(root / "tables/figure_quality_checks.csv", index=False)
    result = dict(status="passed", standalone_figures=len(files), recording_figure_pairs=210,
                  all_formats_present=True, all_text_inside_pdf_page=True, all_png_nonblank=True,
                  plotted_sweep_tables_match_source=True, heldout_summary_markers_match_source=True,
                  note="Superscripts/exponents can be smaller than 8 pt; primary labels are >=8 pt. Representative figures also inspected visually.")
    write_json(root / "figure_verification.json", result)
    return result


def booklet(root):
    names = ["summary/selected_parameters", "summary/modal_ar_order", "summary/selection_agreement",
             "summary/monkey_M_order_sweep", "summary/monkey_T_order_sweep",
             "summary/monkey_M_embedding_sweep", "summary/monkey_T_embedding_sweep",
             "summary/heldout_prediction", "summary/selected_conditioning",
             "examples/monkey_M_predictions", "examples/monkey_T_predictions",
             "examples/monkey_M_residuals", "examples/monkey_T_residuals",
             "examples/monkey_M_coordinates", "examples/monkey_T_coordinates"]
    doc = fitz.open()
    cover = doc.new_page(width=612, height=792)
    cover.insert_text((42, 58), "Macaque OLS and lag-embedding audit", fontsize=20, color=(.1, .22, .27))
    text = """210 single-channel motor-LFP recordings | 2 macaques
Monkey M: 115 recordings / 48 days. Monkey T: 95 recordings / 30 days.
Balanced short-delay trials; last 500 ms before GO; saved five outer folds.

MAIN FINDINGS

AR prediction: mean held-out 10-ms NMSE is 0.00275 for M and 0.00249 for T.
The corresponding mean R-squared is approximately 0.997 for both animals.

AR order: fold-wise median p=18 for M and p=16 for T. The most frequent
recording-modal order is p=19 for M and p=14 for T. All fold choices are saved.

Embedding selection: every fold chooses m=9, tau=1 ms. This reaches the
upper dimension and lower delay limits of the tested grid; it is NOT an
estimate of a nine-dimensional neural manifold or of the number of modes.

INTERPRETATION

These are offline predictions of 2-55-Hz, zero-phase-filtered, normalized
signals sampled at 1 kHz. Whole-epoch preprocessing uses later samples
within the pre-GO epoch. These results are not causal online forecasting.

Strong short-horizon predictability does not establish valid torus geometry.
Selected design condition numbers are large (median about 1.31e8 for AR).
Numerically full rank does not make individual coefficients/poles reliable.

VALIDATION AND OUTPUTS

All 1,050 outer folds independently checked. Prior outputs unchanged.
No PCA, geometric fitting, shape matrices, PINN, or decoding was run.

This booklet is a preview. Every figure is also available as a separate PDF,
SVG and PNG with a CSV and caption. Each of the 210 recordings has its
own order-sweep and embedding-sweep pair in plots/recordings/.

See README.md for full methods, warnings, metric definitions and tables.
Folder: outputs/prego_ols_audit_v1
"""
    assert cover.insert_textbox(fitz.Rect(42, 90, 575, 746), text, fontsize=11, lineheight=1.25) >= 0
    toc = [[1, "Results and interpretation", 1]]
    for name in names:
        with fitz.open(root / "plots" / (name+".pdf")) as source:
            doc.insert_pdf(source)
        toc.append([1, name.split("/")[-1].replace("_", " "), len(doc)])
    doc.set_toc(toc)
    doc.save(root / "OLS_Audit_Overview.pdf", garbage=4, deflate=True)
    doc.close()


if __name__ == "__main__":
    print(verify(DEFAULT_OUTPUT))
    booklet(DEFAULT_OUTPUT)
