"""Render saved envelope-normalized geometry examples without analysis calls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "outputs/stage_hilbert_envelope_tau_sweep_v1"
OUT = ROOT / "geometry_examples"
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")


def boundaries(fit):
    a = np.linspace(0, 2*np.pi, 361)
    c, u, v = (np.asarray(fit[k]) for k in ("center", "u_axis", "v_axis"))
    return [c + x*np.cos(a)[:, None]*u + y*np.sin(a)[:, None]*v
            for x, y in ((fit["R1"], fit["R2"]), (fit["R1_in"], fit["R2_in"]))]


def main():
    OUT.mkdir(exist_ok=True)
    metadata = pd.read_csv(ROOT / "inputs/window_metadata.csv")
    entries = []
    for tau in (21, 23):
        with np.load(ROOT / f"inputs/clouds_tau_{tau:02d}.npz") as saved:
            clouds = saved["clouds"]
            np.testing.assert_array_equal(saved["window_index"], np.arange(1188))
        features = pd.read_csv(ROOT / f"tables/features_native_tau_{tau:02d}.csv").set_index("window_index")
        for trial in (6, 9, 10):
            rows = metadata[metadata.original_trial_number == trial].sort_values("stage")
            assert rows.stage.tolist() == list(range(6))
            for row in rows.itertuples():
                i = int(row.window_index)
                cloud = np.ascontiguousarray(clouds[i])
                result = json.loads((ROOT / f"checkpoints/native/tau_{tau:02d}/window_{i:04d}.json").read_text())
                digest = hashlib.sha256(str((cloud.shape, cloud.dtype.str)).encode()+cloud.tobytes()).hexdigest()
                assert digest == result["cloud_sha256"]
                assert cloud.shape == (300-2*tau, 3) and np.isfinite(cloud).all()
                usable = np.isfinite(features.loc[i].to_numpy()).all()
                entries.append(dict(tau=tau, trial=trial, stage=int(row.stage), cloud=cloud,
                                    boundaries=boundaries(result["fit"]) if usable else [],
                                    usable=bool(usable), window_index=i, cloud_sha256=digest))
    records = []
    for trial in (6, 9, 10):
        selected = [e for e in entries if e["trial"] == trial]
        limit = 1.06*max(np.abs(a).max() for e in selected for a in [e["cloud"], *e["boundaries"]])
        for tau in (21, 23):
            fig = plt.figure(figsize=(9, 5.8), facecolor="white")
            fig.suptitle(f"Original trial {trial} | Delay {tau} ms", fontsize=12, y=.97)
            for stage in range(6):
                e = next(e for e in selected if e["tau"] == tau and e["stage"] == stage)
                ax = fig.add_subplot(2, 3, stage+1, projection="3d", proj_type="ortho")
                ax.plot(*e["cloud"].T, color=COLORS[stage], lw=.65, alpha=.85)
                for line, ls in zip(e["boundaries"], ("-", "--")):
                    ax.plot(*line.T, color="#303030", lw=.9, ls=ls)
                ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
                ax.set_box_aspect((1, 1, 1), zoom=1.15)
                ax.view_init(elev=24, azim=-58)
                ax.set_axis_off()
                ax.set_title(LABELS[stage] + ("\nFit unavailable" if not e["usable"] else ""), fontsize=10, pad=0)
                records.append({k:v for k,v in e.items() if k not in ("cloud", "boundaries")})
            fig.legend(handles=[Line2D([0],[0], color="#71818A", label="Observed trajectory"),
                                Line2D([0],[0], color="#303030", label="Outer boundary"),
                                Line2D([0],[0], color="#303030", ls="--", label="Inner boundary")],
                       loc="lower center", ncol=3, frameon=False, fontsize=8)
            fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.09, wspace=.02, hspace=.13)
            fig.savefig(OUT / f"trial_{trial}_tau_{tau:02d}.png", dpi=600, facecolor="white")
            plt.close(fig)
    pd.DataFrame(records).to_csv(OUT / "examples.csv", index=False)
    (OUT / "README.md").write_text("# Saved geometry examples\n\nTrials 6, 9, and 10; six corrected stages at delays 21 and 23 ms.\n"
        "Observed trajectories and saved annular boundaries; no fitting or decoding rerun.\n"
        "Scales and viewpoints are identical across stages and both delays within each trial.\n"
        "Unusable fits have no outline. Boundaries describe the fitted annular model, not proof of a loop.\n")
    print(OUT)


if __name__ == "__main__":
    main()
