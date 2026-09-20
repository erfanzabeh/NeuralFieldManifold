import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def plotting():
    assert importlib.util.find_spec("plot_prego_panels") is not None
    return importlib.import_module("plot_prego_panels")


def test_requires_frozen_tables_and_rejects_changed_tables(plotting, tmp_path):
    with pytest.raises(FileNotFoundError):
        plotting.load_tables(tmp_path)
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "summary.csv").write_text("changed\n")
    (tmp_path / "frozen_manifest.json").write_text(json.dumps(dict(status="frozen", table_sha256={"summary.csv": "incorrect"})))
    with pytest.raises(ValueError, match="checksum"):
        plotting.load_tables(tmp_path)


def test_selector_accepts_only_planned_panels(plotting):
    assert plotting.selected_panels("C") == ["C"]
    assert plotting.selected_panels("S1") == ["S1"]
    assert plotting.selected_panels("all") == ["A", "B", "C", "D", "E", "F", "S1", "S2"]
    with pytest.raises(ValueError):
        plotting.selected_panels("movement")


def test_plot_module_cannot_import_fit_runner(plotting):
    source = Path(plotting.__file__).read_text()
    for name in ("run_prego_single_channel", "fit_geometry", "fit_elliptical_torus", "motor_lfp_utils"):
        assert name not in source


def test_independent_exports_are_vector_and_text_is_inside_page(plotting, tmp_path):
    import fitz
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    methods = plotting.METHODS + ["torus_power_frequency", "torus_all_bands", "torus_pca"]
    summaries, directions, confusions, nulls, tests, differences, geometry, audit = [], [], [], [], [], [], [], []
    for monkey in "MT":
        for analysis in ("short", "matched_short", "matched_long"):
            for index, method in enumerate(methods):
                summaries.append(dict(monkey=monkey, analysis=analysis, method=method, mean=.2+index*.015,
                                      sd=.035, n_lfps=3, n_days=3, n_trials=270, n_animals=1))
                for d in range(1, 7):
                    directions.append(dict(monkey=monkey, analysis=analysis, method=method, direction=d, mean=.22, std=.04, count=3))
                    for j in range(1, 7):
                        confusions.append(dict(monkey=monkey, analysis=analysis, method=method, true_direction=d,
                                               predicted_direction=j, fraction=.4 if j == d else .12))
        for method in plotting.METHODS:
            nulls.append(dict(monkey=monkey, method=method, mean=.16, low=.15, high=.17, n_permutations=200))
        for panel, enhanced, baseline in (("D", "torus", "power_frequency"), ("D", "torus", "all_bands"),
                                         ("E", "torus_power_frequency", "power_frequency"), ("E", "torus_all_bands", "all_bands")):
            tests.append(dict(monkey=monkey, panel=panel, enhanced=enhanced, baseline=baseline,
                              p_holm=.25, mean_difference=.01, ci_low=-.01, ci_high=.03))
            for i in range(3):
                differences.append(dict(monkey=monkey, enhanced=enhanced, baseline=baseline, recording=f"{monkey}{i}", difference=(i-1)*.03+.01))
        for i in range(90):
            geometry.append(dict(monkey=monkey, recording=f"{monkey}0", direction=i%6+1, fold=i%5,
                                 raw_trial_index=i, R1=rng.uniform(1, 2), R2=rng.uniform(.5, 1), r=rng.uniform(.1, .4)))
        audit.append(dict(monkey=monkey, main_included=True, recording=f"{monkey}0"))
    tables = {"summary": pd.DataFrame(summaries), "direction_summary": pd.DataFrame(directions),
              "mean_confusions": pd.DataFrame(confusions), "null_summary": pd.DataFrame(nulls),
              "planned_comparisons": pd.DataFrame(tests), "paired_differences": pd.DataFrame(differences),
              "example_geometry": pd.DataFrame(geometry), "recording_audit": pd.DataFrame(audit)}
    for panel in plotting.PANELS:
        getattr(plotting, "panel_"+panel.lower())(tables, tmp_path)
    files = list((tmp_path / "panels").glob("*.pdf"))
    assert len(files) == 23
    for path in files:
        assert path.with_suffix(".svg").exists()
        assert path.with_suffix(".png").exists()
        assert path.with_suffix(".csv").exists()
        assert path.with_suffix(".caption.txt").exists()
        with fitz.open(path) as doc:
            assert not doc[0].get_images()
            for block in doc[0].get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                assert doc[0].rect.contains(fitz.Rect(block["bbox"])), (path.name, block["bbox"])
                for line in block["lines"]:
                    assert min(s["size"] for s in line["spans"]) >= 7.99
            if path.name.startswith("D_"):
                labels = doc[0].search_for("frequency")
                assert len(labels) == 2
                labels.sort(key=lambda box: box.x0)
                assert labels[1].x0 - labels[0].x1 >= 3


def test_small_p_values_preserve_threshold_precision(plotting):
    assert plotting.p_text(.001098) == "p = 0.0011"


def test_radius_trial_points_are_visible_above_boxes(plotting):
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame({"direction": [d for d in range(1, 7) for _ in range(3)],
                          "R1": [1., 2., 3.] * 6})
    fig, ax = plt.subplots()
    plotting.radius_plot(ax, frame, "R1")
    assert len(ax.collections) == 6
    assert min(points.get_zorder() for points in ax.collections) > max(box.get_zorder() for box in ax.patches)
    plt.close(fig)
