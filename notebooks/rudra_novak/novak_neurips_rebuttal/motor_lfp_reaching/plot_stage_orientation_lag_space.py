"""Relabel the six saved orientation densities in the lag-coordinate frame."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_stage_geometry_decoding import save, style
from stage_geometry_decoding import COLORS, LABELS, OUTPUT


SOURCE = OUTPUT / "feature_distributions_v1/density_curves.csv"
DESTINATION = OUTPUT.parent / "macaque_task_six_stages_cue_offset_correct/orientation_lag_space_v1"
FEATURES = (("normal_x", "normal_y", "normal_z"),
            ("u_x", "u_y", "u_z"))
ROW_LABELS = (r"Annulus-plane normal, $\mathbf{n}$",
              r"In-plane major-axis direction, $\mathbf{u}$")
COL_LABELS = (r"Lag 0: $x(t)$", r"Lag 3 ms: $x(t-3\,\mathrm{ms})$",
              r"Lag 6 ms: $x(t-6\,\mathrm{ms})$")


def main() -> None:
    curves = pd.read_csv(SOURCE)
    expected = {feature for row in FEATURES for feature in row}
    selected = curves[curves.feature.isin(expected)]
    assert set(selected.feature) == expected
    assert len(selected) == 6 * 6 * 512
    assert not selected.duplicated(["feature", "stage", "x"]).any()

    style()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.1))
    for row, features in enumerate(FEATURES):
        for col, feature in enumerate(features):
            ax = axes[row, col]
            values = selected[selected.feature == feature]
            assert len(values) == 6 * 512
            x = None
            peak = 0.0
            for stage, color in enumerate(COLORS):
                curve = values[values.stage == stage].sort_values("x")
                assert len(curve) == 512
                current_x = curve.x.to_numpy()
                y = curve.density.to_numpy()
                assert np.isfinite(current_x).all() and np.isfinite(y).all()
                assert np.all(y >= 0)
                if x is None:
                    x = current_x
                else:
                    np.testing.assert_allclose(current_x, x, rtol=0, atol=1e-12)
                peak = max(peak, float(y.max()))
                ax.fill_between(x, y, color=color, alpha=0.075, linewidth=0)
                ax.plot(x, y, color=color, lw=1.5)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(0, peak * 1.07)
            ax.set_xlabel("Unit-vector component")
            if col == 0:
                ax.set_ylabel("Density")
            ax.grid(False)
            ax.ticklabel_format(axis="both", useOffset=False, style="plain")
            ax.tick_params(length=3, pad=3)
            if row == 0:
                ax.set_title(COL_LABELS[col], pad=10)

    fig.text(.075, .89, ROW_LABELS[0], fontsize=11, color="#263238")
    fig.text(.075, .415, ROW_LABELS[1], fontsize=11, color="#263238")
    handles = [Line2D([0], [0], color=color, lw=1.8, label=label)
               for label, color in zip(LABELS, COLORS)]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(.5, .985),
               frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.5)
    fig.subplots_adjust(left=.075, right=.955, bottom=.10, top=.76, wspace=.28, hspace=.90)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    exports = []
    save(fig, DESTINATION, "orientation_lag_space_by_stage", exports)
    assert not exports[0]["outside_text"], exports[0]["outside_text"]
    print(DESTINATION / exports[0]["file"])


if __name__ == "__main__":
    main()
