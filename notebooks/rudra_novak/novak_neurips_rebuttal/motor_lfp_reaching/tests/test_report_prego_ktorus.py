from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def report():
    return importlib.import_module("report_prego_ktorus")


def test_metrics_reproduce_predictions_and_normalized_rows(report):
    y = np.repeat(np.arange(1, 7), 10)
    predicted = y.copy()
    predicted[0] = 2
    frame = pd.DataFrame(dict(recording="r", monkey="M", day="day", method="ktorus",
                              raw_trial_index=np.arange(60), true_direction=y,
                              predicted_direction=predicted, fold=np.arange(60)%5,
                              geometry_evaluable=True))
    scores, directions, confusions = report.summarize_predictions(frame)
    assert .97 < scores.macro_f1.iloc[0] < 1
    assert scores.macro_f1.iloc[0] == pytest.approx(directions.f1.mean())
    np.testing.assert_allclose(confusions.groupby("true_direction").fraction.sum(), 1.)
    with pytest.raises(ValueError):
        report.summarize_predictions(pd.concat([frame, frame.iloc[:1]]))


def test_eight_tests_and_paired_differences(report):
    from prego_ktorus import METHODS
    rows = []
    for monkey in ["M", "T"]:
        for recording in range(8):
            for index, method in enumerate(METHODS):
                rows.append(dict(monkey=monkey, recording=f"{monkey}{recording}",
                                 day=f"d{recording//2}", method=method,
                                 macro_f1=.16+.001*recording+.005*index))
    tests, pairs, days = report.planned_statistics(pd.DataFrame(rows))
    assert len(tests) == 8
    assert (tests.p_holm >= tests.p_raw).all()
    assert set(tests.n_lfps) == {8} and set(tests.n_days) == {4}
    for row in tests.itertuples():
        subset = pairs[(pairs.monkey == row.monkey) & (pairs.enhanced == row.enhanced) & (pairs.baseline == row.baseline)]
        assert row.mean_difference == pytest.approx(subset.difference.mean())
        assert row.ci_low <= row.mean_difference <= row.ci_high


def test_table_validation_is_independent_of_cached_scores(report):
    truth = np.repeat(np.arange(1, 7), 10)
    prediction = np.roll(truth, 3)
    score, classes = report.manual_f1(truth, prediction)
    from sklearn.metrics import f1_score
    assert score == pytest.approx(f1_score(truth, prediction, average="macro"))
    np.testing.assert_allclose(classes, f1_score(truth, prediction, average=None))


def test_report_refuses_predecessor_location(report):
    with pytest.raises(ValueError, match="separate experiment"):
        report.freeze(report.PREVIOUS)


def test_report_does_not_contain_geometry_fitting_calls(report):
    import ast
    tree = ast.parse(Path(report.__file__).read_text())
    forbidden = {"fit_trial", "fit_cloud", "learn_geometry", "fit_geometry", "learn_embedding"}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not (calls & forbidden)


def test_coordinate_views_preserve_original_columns(report):
    observed = np.arange(140).reshape(20, 7)
    fitted = observed + .25
    figure = report.coordinate_views(observed, fitted, tau=12, title="Example")
    assert len(figure.axes) == 6
    for j, axis in enumerate(figure.axes, start=1):
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), observed[:, 0])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), observed[:, j])
        np.testing.assert_array_equal(axis.lines[1].get_ydata(), fitted[:, j])
        assert str(j*12) in axis.get_ylabel()
    report.plt.close(figure)


def test_render_only_verifies_frozen_tables(report, tmp_path):
    (tmp_path / "tables").mkdir()
    table = tmp_path / "tables" / "summary.csv"
    table.write_text("mean\n0.2\n")
    report.write_json(tmp_path / "validation.json", dict(table_sha256={"summary.csv": report.sha256(table)}))
    report.verify_frozen_tables(tmp_path)
    table.write_text("mean\n0.3\n")
    with pytest.raises(ValueError, match="Frozen table changed"):
        report.verify_frozen_tables(tmp_path)


def test_examples_reject_changed_source_and_cache(report, tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(report, "CONVERTED_DIR", source)
    folder = tmp_path / "recording"
    folder.mkdir()
    raw = source / "recording.npz"
    raw.write_bytes(b"source")
    cache = folder / "fold_0.npz"
    cache.write_bytes(b"cache")
    report.write_json(folder / "decode_complete.json", dict(source_sha256=report.sha256(raw)))
    report.write_json(folder / "integrity_complete.json", dict(artifacts={cache.name: report.sha256(cache)}))
    report.verify_example_inputs(folder)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Example source changed"):
        report.verify_example_inputs(folder)
    raw.write_bytes(b"source")
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError):
        report.verify_example_inputs(folder)


def test_final_report_archives_source_and_all_tables(report, tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "plots").mkdir()
    report.write_json(tmp_path / "validation.json", {})
    (tmp_path / "tables" / "example_selection.csv").write_text("example\nM\n")
    digest = report.archive_report(tmp_path)
    assert report.sha256(tmp_path / "logs" / "report_source" / f"{digest}.py") == digest
    report.verify_frozen_tables(tmp_path)
    validation = report.json.loads((tmp_path / "validation.json").read_text())
    assert "example_selection.csv" in validation["table_sha256"]


def test_rendered_bar_heights_match_summary(report, tmp_path, monkeypatch):
    from prego_ktorus import METHODS, MAIN_METHODS
    tables = tmp_path / "tables"
    tables.mkdir()
    scores = pd.DataFrame([dict(monkey=m, recording=f"{m}{r}", day=f"d{r//2}",
                                method=method, macro_f1=.16+.002*r+.005*j)
                           for m in ["M", "T"] for r in range(8) for j, method in enumerate(METHODS)])
    summary = scores.groupby(["monkey", "method"]).macro_f1.agg(mean="mean", sd="std").reset_index()
    summary["n_lfps"], summary["n_days"] = 8, 4
    summary.to_csv(tables / "summary.csv", index=False)
    pd.DataFrame([dict(monkey=m, method=method, mean=.15, low=.14, high=.16)
                  for m in ["M", "T"] for method in MAIN_METHODS]).to_csv(tables / "null_summary.csv", index=False)
    tests, pairs, _ = report.planned_statistics(scores)
    tests.to_csv(tables / "planned_comparisons.csv", index=False)
    pairs.to_csv(tables / "paired_differences.csv", index=False)
    pd.DataFrame([dict(monkey=m, method=method, true_direction=i, predicted_direction=j,
                       fraction=.25 if i == j else .15) for m in ["M", "T"]
                  for method in ["power_frequency", "ktorus"] for i in range(1, 7) for j in range(1, 7)]).to_csv(tables / "mean_confusions.csv", index=False)
    pd.DataFrame([dict(monkey=m, method=method, direction=i, mean=.2, std=.04, count=8)
                  for m in ["M", "T"] for method in ["power_frequency", "ktorus"] for i in range(1, 7)]).to_csv(tables / "direction_summary.csv", index=False)
    pd.DataFrame([dict(monkey=m, K=1, dimension=3, tau_samples=14, geometry_evaluable=True)
                  for m in ["M", "T"] for _ in range(5)]).to_csv(tables / "embedding_selection.csv", index=False)
    checked = []

    def inspect(root, name, figure, data, caption):
        figure.canvas.draw()
        if name.startswith("decoding_"):
            axis = figure.axes[0]
            values = data.set_index("method").loc[MAIN_METHODS, "mean"].to_numpy()
            np.testing.assert_allclose([p.get_height() for p in axis.patches], values)
            assert axis.get_ylim()[0] == 0
            checked.append(name)
        if name.startswith("embedding_parameters_"):
            np.testing.assert_array_equal(figure.axes[0].get_xticks(), sorted(data.K.unique()))
            np.testing.assert_array_equal(figure.axes[1].get_xticks(), sorted(data.dimension.unique()))
        assert caption
        report.plt.close(figure)

    monkeypatch.setattr(report, "save_figure", inspect)
    report.render(tmp_path)
    assert checked == ["decoding_M", "decoding_T"]
