import matplotlib.pyplot as plt

color_main = "black"
color_alt = "darkred"

def set_pub_style():
    """Apply publication-quality matplotlib style defaults.

    Configures global ``rcParams`` for clean, journal-ready figures
    with appropriate font sizes, line widths, and minimal spines.
    """
    plt.rcParams.update({
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
    })

def prettify(ax, title=None, xlabel=None, ylabel=None, add_legend=False):
    """Apply unified publication styling to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to style.
    title : str, optional
        Title text for the axes.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    add_legend : bool, optional
        If True, add a frameless legend. Default is False.
    """
    if title: ax.set_title(title, pad=10)
    if xlabel: ax.set_xlabel(xlabel, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, labelpad=5)
    ax.grid(alpha=0.0, linestyle="--", linewidth=0.7)
    if add_legend: 
        ax.legend(frameon=False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.0)