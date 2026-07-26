# Figure style guide

A compact, reproducible style for matplotlib + seaborn figures. It targets clean,
print-ready vector PDF output: serif type, Computer-Modern math, trimmed and offset
spines, endpoint-only ticks, a white background, and a small saturated palette.

Assumed imports for every snippet below:

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
```

## 1. Global setup

Call once before plotting. It sets the theme and rcParams that every figure inherits.

```python
def setup_style():
    sns.set_theme(context="talk", style="ticks", palette="dark")
    plt.rcParams["font.family"] = "serif"          # serif text
    plt.rcParams["mathtext.fontset"] = "cm"        # Computer-Modern math ($...$)
    plt.rcParams["axes.spines.top"] = False        # drop top + right spines
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["lines.linewidth"] = 1            # thin lines
    plt.rcParams["patch.linewidth"] = 0            # no edges on bars/patches
    plt.rcParams["image.interpolation"] = "none"   # crisp heatmaps
    plt.rcParams["legend.frameon"] = False         # no legend box
    plt.rcParams["legend.loc"] = "lower right"
    plt.rcParams["figure.figsize"] = [3.0, 3.0]    # small square default
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.format"] = "pdf"         # vector output
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False
```

## 2. Spines — trimmed and offset

The signature of this style: keep only the left and bottom spines, **trim** each to its
first and last tick, and **offset** it outward so the two spines float and do not meet at
the origin. Apply to every axes after its ticks are set.

```python
sns.despine(ax=ax, trim=True, offset=10)   # single axes
# for a figure of subplots, call it per-axes:
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=10)
```

## 3. Ticks and limits — endpoints only

Label only the extremes. Give a touch of head/tail room on bounded axes so the data
never collides with the spine.

```python
# bounded metric in [0, 1]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 1])

# x runs from 0 to the largest value present
xmax = float(x.max())
ax.set_xlim(0, xmax)
ax.set_xticks([0, xmax])
```

## 4. Typography

- Serif body text; math wrapped in `$...$` renders in Computer-Modern (e.g. `r"Scale $s$"`).
- Titles are short and specific (include the key parameter, e.g. `f"{name}: L{value}"`).
- Axis labels name the quantity, not the column (`"Rate"`, not `"judge_rate"`).
- When rotating tick labels, always set `ha="right"` so labels anchor cleanly to their ticks.

## 5. Lines and colour

Categorical series use a small set of dark, saturated, named colours; reserve **black**
for emphasis and markers. Pick one fixed mapping and keep it consistent across figures.

```python
COLORS = {
    "a": "darkred",
    "b": "midnightblue",
    "c": "darkgreen",
    "derived": "purple",     # e.g. a combined/derived series, drawn dashed
}
EMPHASIS = "black"           # peak markers, reference series

ax.plot(x, y, color=COLORS["a"])
ax.plot(x, z, color=COLORS["derived"], linestyle="--")
ax.plot(x_peak, y_peak, marker="*", color=EMPHASIS,
        markersize=14, fillstyle="none", linestyle="None")
```

## 6. Ordered / sequential data — discrete colormap + colorbar

For an ordered family of lines (a parameter sweep, an index), colour by a **quantized**
`RdYlBu_r` map with one colour per value, and encode the value with a colorbar instead of
a legend.

```python
def custom_cmap(n):
    return ListedColormap(mpl.colormaps["RdYlBu_r"](np.linspace(0, 1, n)))

values = sorted(...)                       # ordered categories, e.g. 0..N
lo, hi = min(values), max(values)
cmap = custom_cmap(len(values))
norm = Normalize(vmin=lo, vmax=hi)

for v in values:
    ax.plot(x, y[v], color=cmap(norm(v)), linewidth=1)

sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cax = fig.add_axes([0.88, 0.30, 0.012, 0.40])   # own axes, right of the plot
cbar = fig.colorbar(sm, cax=cax)
cbar.set_ticks([lo, hi])                          # endpoints only
cbar.set_label("variable", rotation=270, labelpad=12)
```

No legend and no markers on these plots — the colorbar carries the identity.

## 7. Heatmaps

```python
ax = sns.heatmap(matrix, cmap="Greys", square=True, vmin=0, vmax=1,
                 xticklabels=False, yticklabels=False,
                 cbar_kws={"shrink": 0.8, "ticks": [0, 1]})
ax.collections[0].colorbar.ax.set_ylabel("value", rotation=270, labelpad=15)
```

## 8. Legends

Frameless, small, out of the way. Omit entirely when a colorbar already encodes the
series.

```python
ax.legend(loc="best", fontsize=8, frameon=False)
```

## 9. Multi-panel figures

Keep panels square, share one colorbar, and place it in its own axes to the right so it
never shrinks the last panel.

```python
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax in axes:
    # ... plot ...
    ax.set_box_aspect(1)                          # square panels
fig.subplots_adjust(left=0.06, right=0.85, bottom=0.20, top=0.86, wspace=0.45)
for ax in axes:
    sns.despine(ax=ax, trim=True, offset=10)
cax = fig.add_axes([0.88, 0.30, 0.012, 0.40])     # shared colorbar, to the right
```

Give each panel its own short title; put the y-label on the leftmost panel only.

## 10. Saving

Vector PDF, white background, cropped tight.

```python
fig.savefig(path, bbox_inches="tight", facecolor="white", transparent=False)
```
