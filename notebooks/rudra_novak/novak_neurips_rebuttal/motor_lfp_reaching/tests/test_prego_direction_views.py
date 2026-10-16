from pathlib import Path
import ast
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_target_positions_follow_documented_clockwise_order():
    from plot_prego_direction_views import TARGET_XY
    np.testing.assert_allclose(TARGET_XY,
        [[.5, np.sqrt(3)/2], [1, 0], [.5, -np.sqrt(3)/2],
         [-.5, -np.sqrt(3)/2], [-1, 0], [-.5, np.sqrt(3)/2]], atol=1e-14)


def test_display_summaries_are_sample_mean_and_sd_not_fitted_tuning():
    from plot_prego_direction_views import feature_summary
    table = pd.DataFrame({'direction': [1, 1, 1, 2, 2], 'R1': [2., 4., np.nan, 5., 5.]})
    summary = feature_summary(table, ['R1']).set_index('direction')
    assert summary.loc[1, 'n'] == 2
    assert summary.loc[1, 'mean'] == 3
    np.testing.assert_allclose(summary.loc[1, 'sd'], np.sqrt(2))
    assert summary.loc[2, 'mean'] == 5 and summary.loc[2, 'sd'] == 0
    assert summary.loc[3, 'n'] == 0 and np.isnan(summary.loc[3, 'mean'])


def test_target_diagrams_use_saved_rows_and_mark_true_target():
    from plot_prego_direction_views import draw_target_map, CMAP
    fig, ax = plt.subplots()
    row = np.array([0, 1, 2, 3, 4, 5])/15
    draw_target_map(ax, row, 4)
    assert len(ax.patches) == 6
    np.testing.assert_allclose([p.get_facecolor() for p in ax.patches], CMAP(row))
    assert [p.get_linewidth() for p in ax.patches] == [.6, .6, .6, 2, .6, .6]
    for value in row:
        assert f'{value:.2f}' in [t.get_text() for t in ax.texts]
    plt.close(fig)


def test_frozen_inputs_account_for_all_trials_and_valid_fits():
    from plot_prego_direction_views import load_inputs
    features, matrices, counts = load_inputs()
    assert len(features) == 197
    assert features.groupby('direction').size().tolist() == [33, 33, 32, 33, 33, 33]
    for method in ('geometry', 'all_bands'):
        assert counts[method].sum() == 198
        np.testing.assert_array_equal(counts[method].sum(axis=1), [33]*6)
        np.testing.assert_allclose(matrices[method], counts[method]/33)
    assert matrices['geometry'][3, 3] == 0


def test_orientation_view_uses_all_nine_saved_components_and_fixed_scale():
    from plot_prego_direction_views import load_inputs, feature_summary, ORIENTATION, orientation_plot
    features, _, _ = load_inputs()
    summary = feature_summary(features, ORIENTATION)
    fig, ax = orientation_plot(summary)
    image = ax.images[0]
    expected = np.array([[features.loc[features.direction == d, f].mean()
                         for d in range(1, 7)] for f in ORIENTATION])
    np.testing.assert_allclose(image.get_array(), expected)
    assert image.get_clim() == (-1, 1)
    assert len(ax.get_yticklabels()) == 9
    plt.close(fig)


def test_plot_entrypoint_does_not_import_or_call_analysis():
    import plot_prego_direction_views as renderer
    tree = ast.parse(Path(renderer.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(any(s in name for s in ('sklearn', 'NeuralFieldManifold', 'scipy', 'run_prego',
                                         'prego_fixed_geometry_decoding')) for name in imports
                   if name != 'plot_prego_fixed_geometry_decoding')
    calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert not set(calls) & {'fit', 'fit_transform', 'predict', 'predict_proba', 'bootstrap', 'permutation'}
