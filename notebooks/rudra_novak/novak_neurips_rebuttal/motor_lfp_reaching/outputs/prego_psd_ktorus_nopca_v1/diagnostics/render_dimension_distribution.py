"""Render a descriptive diagnostic from frozen results; never refit or decode."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VALIDATION = json.loads((ROOT / "validation.json").read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_artifacts():
    for folder, key in (("tables", "table_sha256"), ("plots", "plot_sha256")):
        for name, expected in VALIDATION[key].items():
            assert sha256(ROOT / folder / name) == expected, name


check_frozen_artifacts()
selection = pd.read_csv(ROOT / "tables/embedding_selection.csv")
audit = pd.read_csv(ROOT / "tables/recording_audit.csv")
fits = pd.read_csv(ROOT / "tables/fit_diagnostics.csv")
selected = selection.loc[selection.geometry_evaluable].copy()
assert not selected.duplicated(["recording", "fold"]).any()
assert (selected.status == "ok").all()
assert (selected.dimension == 2 * selected.K + 1).all()
assert selected.groupby("recording").fold.apply(set).map(lambda x: x == set(range(5))).all()
assert set(selected.recording) == set(audit.loc[audit.geometry_evaluable, "recording"])

rows = []
for monkey, recordings in (("M", 110), ("T", 93)):
    group = selected.loc[selected.monkey == monkey]
    assert group.recording.nunique() == recordings and len(group) == 5 * recordings
    for dimension in (3, 5, 7, 9):
        count = int((group.dimension == dimension).sum())
        rows.append(dict(monkey=monkey, m=dimension, K=(dimension - 1) // 2,
                         count=count, percent=100 * count / len(group),
                         total_folds=len(group), n_recordings=recordings))
counts = pd.DataFrame(rows)
np.testing.assert_allclose(counts.groupby("monkey").percent.sum(), 100)
assert counts["count"].sum() == 1015
counts.to_csv(HERE / "embedding_dimension_distribution.csv", index=False)
per_recording = selected.pivot(index=["monkey", "recording"], columns="fold", values="dimension")
per_recording.columns = [f"m_fold_{fold}" for fold in per_recording.columns]
per_recording["n_distinct_m"] = per_recording.nunique(axis=1)
per_recording.to_csv(HERE / "embedding_dimension_per_recording.csv")

heldout = fits.loc[fits.recording.isin(selected.recording) & fits.held_out & fits.success]
assert not heldout.duplicated(["recording", "raw_trial_index"]).any()
heldout.groupby("monkey").normalized_error.agg(
    successful_heldout_fits="size", median="median", mean="mean"
).to_csv(HERE / "heldout_residual_summary.csv")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 12, "axes.labelsize": 11,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 4.3))
fig.subplots_adjust(left=.105, right=.98, bottom=.24, top=.78)
positions = np.arange(4)
for offset, monkey, color in ((-.18, "M", "#36697f"), (.18, "T", "#b53a32")):
    group = counts.loc[counts.monkey == monkey]
    bars = ax.bar(positions + offset, group.percent, width=.34, color=color,
                  label=f"Monkey {monkey}: {int(group.n_recordings.iloc[0])} LFPs, "
                        f"{int(group.total_folds.iloc[0])} folds")
    np.testing.assert_allclose([bar.get_height() for bar in bars], group.percent)
    for bar, row in zip(bars, group.itertuples()):
        ax.text(bar.get_x() + bar.get_width()/2, row.percent + 1.2,
                f"{row.percent:.1f}%\n(n={row.count})", ha="center", va="bottom", fontsize=9)
ax.set(xticks=positions, xticklabels=["3", "5", "7", "9"], ylim=(0, 74),
       yticks=[0, 20, 40, 60], xlabel="Selected lag-coordinate dimension, m",
       ylabel="Training-fold selections (%)")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#e5e5e5", linewidth=.7)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right", fontsize=9)
fig.text(.105, .935, "Selected embedding dimensions", fontsize=15, weight="bold")
fig.text(.105, .872, "Short-delay trials | 500 ms before GO | No PCA", fontsize=10, color="#555555")
fig.text(.105, .09, "m = 2K + 1, with K estimated from training-only PSD peaks.", fontsize=9)
fig.text(.105, .045, "Five selections per LFP; folds are not independent recordings. Matched decoding cohort only.", fontsize=8)
stem = HERE / "embedding_dimension_distribution"
for extension in ("pdf", "svg", "png"):
    fig.savefig(stem.with_suffix("." + extension), dpi=600, facecolor="white")
plt.close(fig)

caption = (
    "Selected embedding dimensions for the primary matched geometry-decoding cohort: "
    "110 Monkey M LFP recordings (550 outer-training-fold selections) and 93 Monkey T "
    "LFP recordings (465 selections). Each recording contributes five selections. "
    "Bars show the percentage of selections with each m; labels give percentages and counts. "
    "The dimension was set to m=2K+1 from training-only supported PSD peaks, not optimized "
    "against decoding scores. Folds are not independent biological replicates. "
    "All geometry-ineligible recordings, including their otherwise resolved folds, are excluded. "
    "Per-recording dimension choices are provided separately.\n\n"
    "Residual summary: successful held-out trial fits in the same matched cohort, "
    "one fit per recording-trial observation. normalized_error is the time-aligned full-lag "
    "reconstruction SSE divided by the centered total sum of squares. It is not a "
    "nearest-surface distance, a held-out timepoint forecast error, or a decoding error.\n"
)
stem.with_suffix(".txt").write_text(caption)
check_frozen_artifacts()
outputs = [stem.with_suffix("." + ext) for ext in ("csv", "pdf", "svg", "png", "txt")]
outputs += [HERE / "embedding_dimension_per_recording.csv", HERE / "heldout_residual_summary.csv"]
provenance = dict(
    frozen_artifacts_unchanged=True, no_refitting_or_decoding=True,
    source_hashes={name: sha256(ROOT / "tables" / name) for name in (
        "embedding_selection.csv", "recording_audit.csv", "fit_diagnostics.csv")},
    rendering_script_sha256=sha256(Path(__file__)),
    output_hashes={path.name: sha256(path) for path in outputs},
)
(HERE / "embedding_dimension_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(counts.to_string(index=False))
print("Verified cohort, fold counts, percentages, plotted heights, and unchanged frozen results.")
