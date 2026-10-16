"""Stack the saved middle-lag orientation distributions for the slide's left column."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from render_macaque_slide_panels import DENSITIES, export, styling
from stage_geometry_decoding import COLORS, LABELS, OUTPUT


DESTINATION = OUTPUT.parent / "macaque_task_six_stages_cue_offset_correct/slide_panels_v2"


def main() -> None:
    curves = pd.read_csv(DENSITIES)
    wanted = ("normal_y", "u_y")
    curves = curves[curves.feature.isin(wanted)]
    assert len(curves) == 2 * 6 * 512
    styling()
    fig = plt.figure(figsize=(5.4, 4.7))
    axes = (fig.add_axes([.16, .54, .78, .255]),
            fig.add_axes([.16, .115, .78, .255]))
    titles = (r"Annulus-plane normal, $n_2$",
              r"In-plane major-axis direction, $u_2$")
    for ax, feature, title in zip(axes, wanted, titles):
        rows = curves[curves.feature == feature]
        grid = None
        peak = 0.0
        for stage, color in enumerate(COLORS):
            series = rows[rows.stage == stage].sort_values("x")
            assert len(series) == 512
            x = series.x.to_numpy(dtype=float)
            density = series.density.to_numpy(dtype=float)
            assert np.isfinite(x).all() and np.isfinite(density).all()
            assert (density >= 0).all()
            if grid is None:
                grid = x
            else:
                np.testing.assert_allclose(x, grid, rtol=0, atol=1e-12)
            peak = max(peak, float(density.max()))
            ax.fill_between(grid, density, color=color, alpha=.075, linewidth=0)
            ax.plot(grid, density, color=color, lw=1.35)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, peak*1.07)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_ylabel("Density", fontsize=8.5)
        ax.tick_params(length=2.5, labelsize=8)
        ticks = MaxNLocator(4, min_n_ticks=3).tick_values(grid[0], grid[-1])
        ax.set_xticks(ticks[(ticks >= grid[0]) & (ticks <= grid[-1])])
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
        ax.grid(False)
    axes[1].set_xlabel("Unit-vector component", fontsize=8.5)
    fig.text(.08, .98, r"Orientation along $x(t-3\,\mathrm{ms})$",
             va="top", fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], color=color, lw=1.65, label=label)
               for color, label in zip(COLORS, LABELS)]
    fig.legend(handles=handles, ncol=3, loc="upper center", frameon=False,
               bbox_to_anchor=(.51, .935), fontsize=8, handlelength=1.5,
               columnspacing=1.0)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    export(fig, DESTINATION / "geometry_middle_lag_stacked")
    print(DESTINATION / "geometry_middle_lag_stacked.png")


if __name__ == "__main__":
    main()
