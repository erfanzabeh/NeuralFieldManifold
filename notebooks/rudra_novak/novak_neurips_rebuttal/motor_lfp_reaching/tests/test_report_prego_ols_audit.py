import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_recording_summary_weights_trials_not_folds():
    report = importlib.import_module("report_prego_ols_audit")
    rows = pd.DataFrame(dict(recording=["a"]*4, monkey=["M"]*4, day=["d"]*4,
        family=["ar"]*4, horizon_ms=[10]*4, outer_fold=[0, 1, 1, 1],
        nmse=[0., 2., 2., 2.], r2=[1., -1., -1., -1.], nmse_persistence=[3.]*4))
    result = report.recording_scores(rows)
    assert result.iloc[0].nmse == 1.5
    assert result.iloc[0].r2 == -.5


def test_modal_summary_breaks_ties_downwards():
    report = importlib.import_module("report_prego_ols_audit")
    assert report.mode_and_agreement([4, 3, 4, 3, 8]) == (3, .4)


def test_example_recording_uses_only_display_fold_training_scores():
    report = importlib.import_module("report_prego_ols_audit")
    rows = pd.DataFrame(dict(recording=["a", "b", "c", "a", "b", "c"],
        monkey=["M"]*6, family=["ar"]*6, status=["ok"]*6,
        outer_fold=[0, 0, 0, 1, 1, 1], mean_nmse=[1., 2., 4., 50., 999., 60.]))
    assert report.example_recording(rows, "M") == "b"
    rows.loc[rows.outer_fold != 0, "mean_nmse"] = [500., 0., 90.]
    assert report.example_recording(rows, "M") == "b"


def test_independent_metrics_retain_negative_r2():
    report = importlib.import_module("report_prego_ols_audit")
    y = np.array([[0., 1., 2.]])
    metrics = report.independent_metrics(y, np.zeros_like(y), y+1)
    np.testing.assert_allclose(metrics["r2"], [-1.5])
    np.testing.assert_allclose(metrics["nmse_persistence"], [1.5])


def test_independent_metrics_promote_saved_float32_baseline():
    report = importlib.import_module("report_prego_ols_audit")
    rng = np.random.default_rng(13)
    y = rng.normal(size=(20, 290))
    baseline = (y+.1).astype(np.float32)
    a = report.independent_metrics(y, y+.3, baseline)
    b = report.independent_metrics(y, y+.3, baseline.astype(float))
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-13, atol=1e-13)


def test_reporting_never_invokes_analysis_or_fitting():
    import ast
    path = Path(__file__).resolve().parents[1] / "report_prego_ols_audit.py"
    tree = ast.parse(path.read_text())
    forbidden = {"fit_ols", "audit_fold", "process_recording", "fit_geometry", "PCA", "heldout_predictions"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            assert name not in forbidden


def test_order_plot_values_and_log_limits_match_saved_summary(monkeypatch, tmp_path):
    report = importlib.import_module("report_prego_ols_audit")
    rows = []
    for recording, multiplier in (("a", 1.), ("b", 5.)):
        for p in (1, 2):
            for h in (1, 10):
                rows.append(dict(recording=recording, family="ar", p=p, horizon_ms=h,
                    mean_nmse=multiplier*.01/p, mean_persistence=.3, outer_fold_sd=.1))
    frame = pd.DataFrame(rows)
    def capture(fig, path, table, caption):
        for ax in fig.axes:
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [.03, .015])
            np.testing.assert_allclose(ax.lines[1].get_ydata(), [.3, .3])
            assert ax.get_ylim()[0] > 1e-6
        report.plt.close(fig)
    monkeypatch.setattr(report, "export", capture)
    report.order_plot(frame, tmp_path / "test", "Test", population=True)


def test_heatmap_masks_unavailable_cells_without_changing_values(monkeypatch, tmp_path):
    report = importlib.import_module("report_prego_ols_audit")
    rows = []
    for h in (1, 10):
        for m, tau, error in ((2, 1, .2), (2, 100, .3), (4, 1, .1)):
            rows.append(dict(family="embedding", m=m, tau=tau, horizon_ms=h, mean_nmse=error))
    def capture(fig, path, table, caption):
        for ax in [a for a in fig.axes if a.images]:
            data = ax.images[0].get_array()
            np.testing.assert_allclose(data[0], [.2, .3])
            assert data.mask[1, 1]
        report.plt.close(fig)
    monkeypatch.setattr(report, "export", capture)
    report.embedding_plot(pd.DataFrame(rows), tmp_path / "test", "Test", dict(dimensions=[2, 4], delays=[1, 100]))
