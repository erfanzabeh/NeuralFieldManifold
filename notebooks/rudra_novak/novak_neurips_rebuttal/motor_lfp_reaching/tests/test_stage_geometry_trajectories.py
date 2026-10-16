from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_stage_geometry_trajectories import trajectory_axis
from plot_stage_geometry_decoding import outlines
from stage_geometry_decoding import COLORS
import matplotlib.pyplot as plt


@pytest.mark.parametrize("overlay", [False, True])
def test_only_point_rendering_changes(overlay):
    cloud = np.random.default_rng(23).normal(size=(294, 3))
    result = dict(usable=True, fit=dict(center=[0, 0, 0], u_axis=[1, 0, 0], v_axis=[0, 1, 0],
                                      R1=3., R2=2., R1_in=1.5, R2_in=1.))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    trajectory_axis(ax, cloud, result, 3, 5., overlay)
    assert len(ax.collections) == 0
    assert len(ax.lines) == (3 if overlay else 1)
    observed = ax.lines[0]
    np.testing.assert_array_equal(np.column_stack(observed.get_data_3d()), cloud)
    assert observed.get_color() == COLORS[3]
    assert observed.get_linewidth() == 1. and observed.get_alpha() == 1.
    assert ax.elev == 24 and ax.azim == -58
    for limits in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()):
        np.testing.assert_array_equal(limits, [-5, 5])
    if overlay:
        for line, expected in zip(ax.lines[1:], outlines(result)):
            np.testing.assert_array_equal(np.column_stack(line.get_data_3d()), expected)
    plt.close(fig)
