import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_population_summary_keeps_flagged_and_nonconverged_results():
    report = importlib.import_module("report_prego_geometric_fits")
    table = pd.DataFrame(dict(model=["one_torus"]*4, status=["returned"]*3+["failed"],
                              optimizer_success=[True, False, True, False],
                              flags=["", "nonconvergence", "near_optimizer_bound", "fit_failed"],
                              normalized_3d_set_error=[.1, .2, .9, np.nan]))
    summary = report.summarize(table)
    row = summary.iloc[0]
    assert row.n_results == 4
    assert row.n_finite == 3
    assert row.n_converged == 2
    assert row.n_flagged == 3
    assert row.n_collapsed_hole == 0
    assert row.n_nonconverged == 1
    assert row.n_bound_hits == 1
    assert row["median"] == .2
    assert row.q25 == pytest.approx(.15)
    assert row.q75 == pytest.approx(.55)


def test_rendered_mesh_respects_fitted_plane_and_tube():
    report = importlib.import_module("report_prego_geometric_fits")
    f = dict(center=[0., 0., 0.], direction=[0., 0., 1.], u_axis=[1., 0., 0.],
             v_axis=[0., 1., 0.], R1=2., R2=2., R1_in=1., R2_in=1., minor_radius=.25)
    band = report.model_mesh(f, "one_torus", 33)
    assert band.shape[-1] == 3
    np.testing.assert_allclose(band[..., 2], 0)
    radius = np.linalg.norm(band[..., :2], axis=-1)
    assert radius.min() == pytest.approx(1.)
    assert radius.max() == pytest.approx(2.)
    tube = report.model_mesh(f, "two_torus", 33)
    distance = np.sqrt((np.linalg.norm(tube[..., :2], axis=-1)-2)**2+tube[..., 2]**2)
    np.testing.assert_allclose(distance, .25, atol=1e-12)


def test_report_is_plot_only():
    import ast
    source = Path(__file__).resolve().parents[1]/"report_prego_geometric_fits.py"
    tree = ast.parse(source.read_text())
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name)}
    assert not called.intersection({"fit_one", "one_torus_fit", "two_torus_fit", "decode"})


def test_long_diagnostics_fit_inside_each_trial_panel():
    import matplotlib.pyplot as plt
    report = importlib.import_module("report_prego_geometric_fits")
    core = importlib.import_module("prego_geometric_fits")
    t = np.linspace(0, 2*np.pi, 40, endpoint=False)
    p = np.c_[2*np.cos(t), np.sin(t), .2*np.sin(3*t)]
    results = {model: core.fit_one(p, model) for model in core.MODELS}
    for result in results.values():
        result["summary"]["flags"] = "nonconvergence;collapsed_hole;active_optimizer_bound;near_optimizer_bound;nearly_circular_orientation"
    fig = report.pair_figure(p, results, pd.Series(dict(direction=1, original_trial_number=16)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        box = text.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert box.x0 >= 0 and box.x1 <= 1
        assert box.y0 >= 0 and box.y1 <= 1
        if "Flags:" in text.get_text() and text.get_position()[0] < .5:
            assert box.x1 < .52
    plt.close(fig)


def test_thick_tube_display_warns_about_sweep_not_volume_boundary():
    report = importlib.import_module("report_prego_geometric_fits")
    result = dict(status="returned", summary=dict(flags="collapsed_hole;nonconvergence"))
    note = report.display_note(result, "two_torus")
    assert "collapsed hole" in note.lower()
    assert "not converged" in note.lower()
    assert "parametric" in note.lower()
    assert "not a volume boundary" in note.lower()
    fit = dict(center=[0, 0, 0], direction=[0, 0, 1], u_axis=[1, 0, 0],
               v_axis=[0, 1, 0], R1=2., R2=2., minor_radius=3.)
    sweep = report.model_mesh(fit, "two_torus", 33)
    backbone_distance = np.sqrt((np.linalg.norm(sweep[..., :2], axis=-1)-2)**2+sweep[..., 2]**2)
    assert backbone_distance.min() < fit["minor_radius"]-1


@pytest.mark.parametrize("field,value", [("flags", "tampered"), ("optimizer_success", False),
                                          ("R1", 900.), ("original_trial_number", 999999)])
def test_export_validation_rejects_parameter_flag_and_identity_changes(field, value):
    report = importlib.import_module("report_prego_geometric_fits")
    core = importlib.import_module("prego_geometric_fits")
    t = np.linspace(0, 2*np.pi, 60, endpoint=False)
    cloud = np.c_[2*np.cos(t), np.sin(t), .1*np.sin(4*t)]
    result = core.fit_one(cloud, "one_torus")
    trial = dict(row_index=0, original_trial_number=16, raw_trial_index=6,
                 direction=1, heldout_fold_zero_based=4, delay="short")
    result.update({k: trial[k] for k in ("row_index", "original_trial_number", "raw_trial_index", "direction")})
    exported = dict(trial, status=result["status"], reason=result["reason"],
                    cloud_sha256=result["cloud_sha256"], **result["summary"])
    report.verify_export_row(exported, result, trial, cloud)
    exported[field] = value
    with pytest.raises((ValueError, AssertionError)):
        report.verify_export_row(exported, result, trial, cloud)
