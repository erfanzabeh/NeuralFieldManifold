import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_saved_fit_is_bound_to_actual_full_coordinate_cloud():
    module = importlib.import_module("complete_prego_ktorus")
    from prego_ktorus import fit_trial
    t = np.arange(500)/1000
    x = np.sin(2*np.pi*17*t)+.7*np.cos(2*np.pi*33*t)
    embedding = dict(K=2, dimension=5, tau=15, frequencies=[17., 33.], status="ok")
    features, result = fit_trial(x, embedding)
    assert result["success"]
    module.check_saved_trial(x, embedding, result["coefficients"], result["frequencies"], features)
    with pytest.raises(AssertionError):
        module.check_saved_trial(x, dict(embedding, tau=21), result["coefficients"], result["frequencies"], features)


def test_pending_nulls_cover_missing_fused_comparisons():
    module = importlib.import_module("complete_prego_ktorus")
    import pandas as pd
    from prego_ktorus import MAIN_METHODS, METHODS
    frame = pd.DataFrame([dict(method=m, permutation=p) for m in MAIN_METHODS for p in range(200)])
    pending = module.pending_nulls(frame, 200)
    assert len(pending) == 200
    assert all(set(methods) == set(METHODS)-set(MAIN_METHODS) for methods in pending.values())
