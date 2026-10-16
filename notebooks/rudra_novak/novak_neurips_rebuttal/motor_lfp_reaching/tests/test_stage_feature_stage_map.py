"""Check descriptive stage fingerprint scaling and source accounting."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_stage_feature_stage_map as stage_map


def test_source_accounting_and_robust_scaling():
    table = pd.read_csv(stage_map.SOURCE, float_precision="round_trip")
    medians, shifts, center, scale, counts = stage_map.summarize(table)
    assert shifts.shape == (6, 15)
    assert counts.sum() == 1177
    assert counts.tolist() == [197, 198, 196, 195, 196, 195]
    np.testing.assert_allclose(shifts.to_numpy(),
                               medians.sub(center).div(scale).to_numpy(), atol=1e-12)
    assert np.isfinite(shifts.to_numpy()).all()
    assert shifts.loc["post_GO", "R2"] < -.8
    assert shifts.loc["pre_TC", "R2"] > 0


def test_source_order_and_labels():
    assert len(stage_map.FEATURES) == len(stage_map.FEATURE_LABELS) == 15
    assert stage_map.FEATURES[:3] == ("R1", "R2", "band_half_width")
    assert stage_map.FEATURES[-3:] == ("mse", "mean_error", "frac_inside")
