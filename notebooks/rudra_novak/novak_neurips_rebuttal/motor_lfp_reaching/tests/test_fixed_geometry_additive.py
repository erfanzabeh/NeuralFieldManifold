from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_six_representations_use_frozen_beta_and_geometry_without_width():
    from run_prego_fixed_geometry_additive import build_features, METHODS
    bands = np.arange(40).reshape(8, 5)
    geo = np.arange(112).reshape(8, 14)
    avg = np.arange(8)[:, None]
    features = build_features(bands, geo, avg)
    assert tuple(features) == METHODS
    assert [v.shape[1] for v in features.values()] == [1, 15, 1, 15, 5, 14]
    np.testing.assert_array_equal(features['relevant_band'], bands[:, 3:4])
    np.testing.assert_array_equal(features['geometry_relevant_band'], np.c_[geo, bands[:, 3]])
    np.testing.assert_array_equal(features['geometry_average_psd'], np.c_[geo, avg])
    np.testing.assert_array_equal(features['all_bands'], bands)
    np.testing.assert_array_equal(features['geometry'], geo)


def test_average_psd_is_mean_density_not_mean_of_log_bands():
    from run_prego_fixed_geometry_additive import average_psd
    t = np.arange(500)/1000
    x = np.stack([np.sin(2*np.pi*22*t), 2*np.sin(2*np.pi*22*t)])
    f, p = signal.welch(signal.detrend(x, axis=1), fs=1000, window='hann', nperseg=500,
                        noverlap=250, nfft=10000, detrend=False, axis=1)
    expected = np.log10(np.maximum(p[:, (f >= 2)&(f <= 55)].mean(axis=1), 1e-30))
    np.testing.assert_allclose(average_psd(x)[:, 0], expected)
    np.testing.assert_allclose(np.diff(average_psd(x)[:, 0]), np.log10(4))


def test_projection_training_axes_do_not_use_heldout_features_or_labels():
    from run_prego_fixed_geometry_additive import fit_projection
    rng = np.random.default_rng(77)
    y = np.repeat(np.arange(1, 7), 15)
    folds = np.tile(np.arange(5), 18)
    x = rng.normal(size=(90, 14))
    x[4] = np.nan
    a, model_a, sign_a, mean_a, scale_a = fit_projection(x, y, folds)
    changed = x.copy()
    changed[folds == 0] = 1000
    other_y = y.copy()
    other_y[folds == 0] = np.roll(other_y[folds == 0], 1)
    b, model_b, sign_b, mean_b, scale_b = fit_projection(changed, other_y, folds)
    np.testing.assert_array_equal(a[folds != 0], b[folds != 0])
    np.testing.assert_array_equal(model_a['lda'].scalings_, model_b['lda'].scalings_)
    np.testing.assert_array_equal(sign_a, sign_b)
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(scale_a, scale_b)
    assert a.shape == (90, 2)
    assert list(model_a.named_steps) == ['impute', 'scale', 'lda']


def test_comparison_has_only_one_horizontal_reference_and_no_grid():
    from plot_prego_fixed_geometry_additive import comparison_plot
    from run_prego_fixed_geometry_additive import METHODS
    table = pd.DataFrame(dict(method=METHODS, macro_f1=np.linspace(.1, .3, 6),
                              ci_low=np.linspace(.05, .25, 6), ci_high=np.linspace(.15, .35, 6)))
    fig, ax = comparison_plot(table)
    assert len(ax.patches) == 6
    assert not any(line.get_visible() for line in ax.get_xgridlines()+ax.get_ygridlines())
    lines = [line for line in ax.lines if line.get_linestyle() == '--']
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0].get_ydata(), 1/6)
    assert not any('shuffl' in t.get_text().lower() for t in ax.texts)
    plt.close(fig)


def test_density_contours_use_training_trials_only():
    from plot_prego_fixed_geometry_additive import density_arrays
    rng = np.random.default_rng(4)
    xy = rng.normal(size=(120, 2))
    y = np.tile(np.arange(1, 7), 20)
    folds = np.repeat(np.arange(5), 24)
    first = density_arrays(xy, y, folds)
    changed = xy.copy()
    changed[folds == 0] = 999
    second = density_arrays(changed, y, folds)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
