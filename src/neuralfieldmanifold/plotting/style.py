"""Shared figure styling for the tutorial notebooks."""


def set_style():
    """Apply the tutorial typography, colors, and line defaults."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def prettify(ax, title=None, xlabel=None, ylabel=None, add_legend=False):
    """Set axis labels, remove grid lines, and optionally add a legend."""
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=5)
    ax.grid(alpha=0.0, linestyle="--", linewidth=0.7)
    if add_legend:
        ax.legend(frameon=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)
