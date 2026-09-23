"""Fixed six-stage geometry experiment; original-coordinate clouds and trial CV."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from convert_motor_lfp import h5_cell_dataset, h5_cell_array, trial_column
from motor_lfp_utils import detrend_zscore, bandpass_filter, lag_embed
from prego_decoding import make_estimator
from prego_fixed_geometry_decoding import geometry_features as features14

UNIT = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
PREVIOUS = UNIT / "outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1"
OUTPUT = UNIT / "outputs/stage_T_y070316009_12_ch7_K1_m3_tau3_300ms_15D_v1"
STAGES = ("pre_TC", "post_TC", "pre_SC", "post_SC", "pre_GO", "post_GO")
LABELS = ("Pre-TC", "Post-TC", "Pre-SC", "Post-SC", "Pre-GO", "Post-GO")
COLORS = ("#9AB7D3", "#376D9B", "#E8B1A3", "#BE5A47", "#9DC5B5", "#38816A")
COLUMNS = ["R1", "R2", "band_half_width", "mse", "mean_error", "frac_inside",
           "normal_x", "normal_y", "normal_z", "u_x", "u_y", "u_z", "v_x", "v_y", "v_z"]


def stage_intervals(event_samples):
    events = np.asarray(event_samples)
    if events.shape != (3,) or not np.isfinite(events).all() or not np.equal(events, events.astype(int)).all():
        raise ValueError("Expected integer zero-based TC, SC, GO onsets")
    intervals = np.array([(e-300, e) if pre else (e, e+300)
                          for e in events.astype(int) for pre in (True, False)])
    if np.any(intervals[:-1, 1] > intervals[1:, 0]):
        raise ValueError("Stage windows overlap")
    return intervals


def load_raw_windows():
    """Audit raw event codes and extract half-open slices without inferred delays."""
    trials = pd.read_csv(PREVIOUS / "inputs/selected_trials.csv")
    if len(trials) != 198 or trials.original_trial_number.nunique() != 198:
        raise ValueError("Saved cohort changed")
    np.testing.assert_array_equal(trials.groupby("direction").size(), np.full(6, 33))
    rows, windows, event_rows = [], [], []
    with np.load(UNIT / "lfp_converted_data" / f"{RECORDING}.npz", allow_pickle=True) as z, \
            h5py.File(UNIT / "raw_data/MonkeyT.mat", "r") as f:
        if int(z["sampling_rate_hz"]) != 1000:
            raise ValueError("Expected 1000 Hz")
        source_row = int(z["source_pair_index"])
        group = f["Monkey"]
        codes_ds = h5_cell_dataset(f, group, "TrialCodesCorr", source_row)
        times_ds = h5_cell_dataset(f, group, "TrialTimesCorr", source_row)
        cache = {}
        for trial_index, row in enumerate(trials.to_dict("records")):
            i = int(row["raw_trial_index"])
            if int(z["original_trial_number"][i]) != row["original_trial_number"]:
                raise ValueError("Trial identity mismatch")
            if str(z["delay_label"][i]) != "short" or int(z["direction"][i]) != row["direction"]:
                raise ValueError("Trial condition mismatch")
            ci, ti = int(z["source_condition_index"][i])-1, int(z["source_trial_index"][i])-1
            if ci not in cache:
                cache[ci] = (h5_cell_array(f, codes_ds, ci), h5_cell_array(f, times_ds, ci))
            codes, times = [trial_column(a, ti) for a in cache[ci]]
            if 91 not in codes or 92 in codes:
                raise ValueError("Raw code does not identify a short-delay trial")
            events = []
            for code in (202, 204, 207):
                hit = np.flatnonzero(codes == code)
                if len(hit) != 1:
                    raise ValueError(f"Trial {row['original_trial_number']}: event {code} ambiguous")
                t = times[hit[0]]
                if not np.isfinite(t) or not float(t).is_integer():
                    raise ValueError("Invalid raw event time")
                events.append(int(t)-1)
            if events[2] != 4500:
                raise ValueError("GO alignment changed")
            event_rows.append(dict(trial_index=trial_index, original_trial_number=row["original_trial_number"],
                                   TC_sample=events[0], SC_sample=events[1], GO_sample=events[2],
                                   source_condition_index=ci+1, source_trial_index=ti+1))
            for stage, (start, end) in enumerate(stage_intervals(events)):
                x = z["lfp"][i, start:end]
                if start < 0 or end > z["lfp"].shape[1] or x.shape != (300,):
                    raise ValueError("Incomplete stage window")
                if not np.isfinite(x).all() or np.ptp(x) <= 1e-12:
                    raise ValueError("Invalid raw stage window")
                rows.append(dict(row, window_index=len(rows), trial_index=trial_index,
                                 stage=stage, stage_name=STAGES[stage], start_sample=int(start),
                                 end_sample_exclusive=int(end), event_sample=events[stage//2]))
                windows.append(x)
    return trials, pd.DataFrame(rows), pd.DataFrame(event_rows), np.stack(windows)


def preprocess_window(raw):
    raw = np.asarray(raw)
    if raw.shape != (300,) or not np.isfinite(raw).all() or np.ptp(raw) <= 1e-12:
        raise ValueError("Expected a finite, nonconstant 300-sample window")
    normalized = detrend_zscore(raw)
    processed = bandpass_filter(normalized, fs=1000, low=2., high=55., order=4).astype(np.float32)
    if not np.isfinite(processed).all() or np.ptp(processed) <= 1e-12:
        raise ValueError("Degenerate processed window")
    return processed


def embed_window(processed):
    if np.asarray(processed).shape != (300,) or not np.isfinite(processed).all():
        raise ValueError("Invalid processed window")
    cloud = lag_embed(processed, dim=3, tau=3)
    np.testing.assert_array_equal(cloud, np.column_stack((processed[6:], processed[3:-3], processed[:-6])))
    return cloud


def geometry_features(result):
    old, reason = features14(result)
    if reason:
        return np.full(15, np.nan), reason
    fit = result["fit"]
    expected = (fit["R1"]-fit["R1_in"]+fit["R2"]-fit["R2_in"])/4
    width = fit["minor_radius"]
    if not np.isfinite(width) or width <= 0 or not np.isclose(width, expected, rtol=1e-10, atol=1e-12):
        return np.full(15, np.nan), "invalid_band_half_width"
    return np.insert(old, 2, width), ""


def validate_groups(labels, folds, trial_ids):
    labels, folds, trial_ids = map(np.asarray, (labels, folds, trial_ids))
    if not (len(labels) == len(folds) == len(trial_ids)) or set(np.unique(folds)) != set(range(5)):
        raise ValueError("Expected matching rows and five saved folds")
    for trial in np.unique(trial_ids):
        mask = trial_ids == trial
        if np.unique(folds[mask]).size != 1:
            raise ValueError("Trial windows cross folds")
        if not np.array_equal(np.sort(labels[mask]), np.arange(6)):
            raise ValueError("Every trial must contain six stages")


def decode(x, labels, folds, trial_ids, retain_models=False):
    x, labels, folds = np.asarray(x), np.asarray(labels), np.asarray(folds)
    validate_groups(labels, folds, trial_ids)
    if x.shape != (len(labels), 15) or np.isinf(x).any():
        raise ValueError("Expected 15 finite-or-missing features")
    pred = np.full(len(labels), -1, dtype=np.int8)
    models = {}
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if not np.isfinite(x[train]).any(axis=0).all():
            raise ValueError(f"No usable training values in fold {fold}")
        model = make_estimator()
        assert list(model.named_steps) == ["impute", "scale", "lda"]
        model.fit(x[train], labels[train])
        pred[test] = model.predict(x[test])
        if retain_models:
            models[fold] = model
    assert np.isin(pred, np.arange(6)).all()
    return pred, models


def scores(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    if y.shape != p.shape or y.ndim != 1 or not np.isin(y, np.arange(6)).all() or not np.isin(p, np.arange(6)).all():
        raise ValueError("Invalid stage predictions")
    counts = np.bincount(y*6+p, minlength=36).reshape(6, 6)
    tp = counts.diagonal()
    denom = counts.sum(axis=0)+counts.sum(axis=1)
    f1 = np.divide(2*tp, denom, out=np.zeros(6), where=denom > 0)
    totals = counts.sum(axis=1, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros((6, 6)), where=totals > 0)
    return dict(macro_f1=f1.mean(), accuracy=tp.sum()/len(y), per_stage_f1=f1,
                confusion_counts=counts, confusion_fraction=fractions)


def shuffle_stages(labels, trial_ids, index, seed=52000):
    rng = np.random.default_rng(seed+index)
    shuffled = np.array(labels, copy=True)
    for trial in np.unique(trial_ids):
        mask = trial_ids == trial
        shuffled[mask] = rng.permutation(shuffled[mask])
    return shuffled


def permutation(index, x, labels, folds, trial_ids):
    shuffled = shuffle_stages(labels, trial_ids, index)
    predictions, _ = decode(x, shuffled, folds, trial_ids)
    s = scores(shuffled, predictions)
    return dict(index=index, labels=shuffled, predictions=predictions,
                macro_f1=s["macro_f1"], accuracy=s["accuracy"])


def bootstrap(labels, predictions, trial_ids, reach_direction, count=2000, seed=42000):
    rng = np.random.default_rng(seed)
    ids = np.unique(trial_ids)
    groups = {t: np.flatnonzero(trial_ids == t) for t in ids}
    strata = [[t for t in ids if reach_direction[groups[t][0]] == d] for d in range(1, 7)]
    samples = np.stack([np.concatenate([rng.choice(g, len(g), replace=True) for g in strata])
                        for _ in range(count)]).astype(int)
    result = [scores(labels[idx], predictions[idx]) for sample in samples
              for idx in [np.concatenate([groups[t] for t in sample])]]
    return dict(trial_indices=samples, **{k: np.array([s[k] for s in result])
                for k in ("macro_f1", "accuracy", "per_stage_f1")})
