from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_stage_hilbert_sweep as experiment
import run_stage_tau_sweep as sweep
from plot_stage_normalization_geometry import envelope_window, trial_envelope
from stage_geometry_decoding import shuffle_stages


def test_delay_indices_and_no_previous_sweep_defaults_changed(monkeypatch):
    old_taus, old_variants = sweep.TAUS, sweep.VARIANTS
    monkeypatch.setattr(sweep, "TAUS", old_taus)
    monkeypatch.setattr(sweep, "VARIANTS", old_variants)
    monkeypatch.setattr(sweep, "PILOT_TAUS", sweep.PILOT_TAUS)
    experiment.configure_sweep(range(1, 101))
    signal = np.arange(300, dtype=float)
    for tau in range(1, 101):
        points = sweep.cloud(signal, tau, "native")
        assert points.shape == (300 - 2*tau, 3)
        np.testing.assert_array_equal(points[0], [2*tau, tau, 0])
        np.testing.assert_array_equal(points[-1], [299, 299-tau, 299-2*tau])
    assert sweep.ROOT != experiment.ROOT


def test_envelope_gain_behavior():
    t = np.arange(6301) / 1000
    signal = (2 + .1 * np.sin(2*np.pi*t)) * np.sin(2*np.pi*22*t)
    _, env, scale = trial_envelope(signal)
    _, larger_env, larger_scale = trial_envelope(signal * 3)
    np.testing.assert_allclose(larger_env, env*3, atol=1e-8)
    base = envelope_window(signal[2000:2300], env[2000:2300], scale)
    larger = envelope_window(signal[2000:2300]*3, larger_env[2000:2300], larger_scale)
    np.testing.assert_allclose(larger, base*3, rtol=2e-6, atol=1e-6)


def test_train_only_imputation_and_scaling():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 15))
    x[0] = np.nan
    labels = np.tile(np.arange(6), 20)
    folds = np.repeat(np.arange(20) % 5, 6)
    _, models = experiment.generic_decode(x, labels, folds)
    changed = x.copy()
    changed[folds == 0] += 100
    _, altered_models = experiment.generic_decode(changed, labels, folds)
    for step in ("impute", "scale"):
        attribute = "statistics_" if step == "impute" else "mean_"
        np.testing.assert_array_equal(getattr(models[0].named_steps[step], attribute),
                                      getattr(altered_models[0].named_steps[step], attribute))


def test_permutations_keep_stage_balance_per_trial():
    labels = np.tile(np.arange(6), 198)
    ids = np.repeat(np.arange(198), 6)
    shuffled = shuffle_stages(labels, ids, 3)
    np.testing.assert_array_equal(np.sort(shuffled.reshape(198, 6), axis=1), labels.reshape(198, 6))


def test_plot_only_does_not_call_analysis(monkeypatch):
    import plot_stage_hilbert_sweep as plots
    for function in ("fit", "calculate_ami", "selected_reports", "generic_decode"):
        monkeypatch.setattr(experiment, function, lambda *a, **k: pytest.fail("Plot-only invoked analysis"))
    assert plots.ROOT == experiment.ROOT


def test_frozen_example_window_join():
    metadata = pd.read_csv(experiment.SOURCE / "inputs/window_metadata.csv")
    examples = pd.read_csv(experiment.EXAMPLES / "windows.csv")
    selected = metadata.set_index("window_index").loc[examples.window_index]
    for name in ("original_trial_number", "stage", "direction", "start_sample", "end_sample_exclusive"):
        np.testing.assert_array_equal(selected[name], examples[name])


def test_spawned_worker_accepts_tau25(tmp_path):
    from joblib import Parallel, delayed, parallel_config
    row = dict(window_index=0, original_trial_number=6, stage_name="pre_TC")
    path = tmp_path / "fit.json"
    # A nonfinite cloud fails explicitly inside the fitter, not in delay validation.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        result = Parallel(n_jobs=2)([delayed(experiment.fit_worker)(path, row, np.full(300, np.nan), 25)])
    assert result[0][0] == "failed"
    import json
    saved = json.loads(path.read_text())
    assert saved["tau_ms"] == 25 and saved["point_count"] == 250
    assert "finite Nx3" in saved["reason"]


def test_plot_only_renders_without_analysis_or_clipped_text(tmp_path, monkeypatch):
    import json
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    import plot_stage_hilbert_sweep as plots
    from run_stage_additive_spectral import METHODS
    from stage_geometry_decoding import STAGES
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    for function in ("fit", "calculate_ami", "selected_reports", "generic_decode"):
        monkeypatch.setattr(experiment, function, lambda *a, **k: pytest.fail("Plot-only invoked analysis"))
    (tmp_path / "tables").mkdir()
    (tmp_path / "selection.json").write_text(json.dumps(dict(f1_tau_ms=1, f1=1., ami_tau_ms=3)))
    pd.DataFrame([dict(tau_ms=tau, stage_name=stage, f1=1.) for tau in range(1,26) for stage in STAGES]).to_csv(tmp_path / "tables/stage_f1.csv", index=False)
    pd.DataFrame(dict(tau_ms=np.arange(1,26), macro_f1=1.)).to_csv(tmp_path / "tables/overall_f1.csv", index=False)
    pd.DataFrame(dict(scope="pooled", bins=48, tau_ms=np.arange(1,101), ami_nats=(np.arange(1,101)-3.)**2, smoothed_ami_nats=(np.arange(1,101)-3.)**2)).to_csv(tmp_path / "tables/ami_curves.csv", index=False)
    labels = np.tile(np.arange(6), 198)
    for tau in (1,3):
        out = tmp_path / "selected" / f"tau_{tau:02d}"
        out.mkdir(parents=True)
        pd.DataFrame(np.eye(6), index=STAGES, columns=STAGES).to_csv(out / "recall_geometry.csv")
        np.savez_compressed(out / "predictions.npz", labels=labels, predictions=np.tile(labels[:,None], (1,7)))
        pd.DataFrame(dict(stage=STAGES, f1=1., ci_low=1., ci_high=1.)).to_csv(out / "per_stage_f1.csv", index=False)
        pd.DataFrame(dict(method=METHODS, macro_f1=1., ci_low=1., ci_high=1.)).to_csv(out / "summary.csv", index=False)
        np.savez_compressed(out / "bootstrap.npz", macro_f1=np.ones((2000,7)))
        (out / "metrics.json").write_text(json.dumps(dict(permutation_p=.001, sensitivity=dict(complete_trials=198))))
        pd.DataFrame(dict(stage=STAGES, unusable=0)).to_csv(out / "fit_accounting.csv", index=False)
    captured = []
    def checked_save(fig, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boundary = fig.bbox
        excluded = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not low <= tick.get_loc() <= high:
                        excluded.update((tick.label1, tick.label2))
        for text in fig.findobj(Text):
            if text in excluded or not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            assert box.x0 >= boundary.x0-2 and box.x1 <= boundary.x1+2, (name, text.get_text())
            assert box.y0 >= boundary.y0-2 and box.y1 <= boundary.y1+2, (name, text.get_text())
        fig.savefig(tmp_path / name, dpi=80)
        captured.append(name)
        plt.close(fig)
    monkeypatch.setattr(plots, "save", checked_save)
    plots.render()
    assert len(captured) == 8
    assert (tmp_path / "REPORT.md").exists()
