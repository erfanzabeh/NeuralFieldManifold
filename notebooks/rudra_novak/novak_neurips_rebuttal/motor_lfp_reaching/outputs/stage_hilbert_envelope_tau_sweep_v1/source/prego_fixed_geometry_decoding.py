"""Fixed 14D annular-band decoding; no embedding or model selection."""
from __future__ import annotations

import numpy as np

from prego_decoding import make_estimator, pack_torus, permute_within_folds
from prego_geometric_fits import UNIT

OUTPUT = UNIT / "outputs/prego_T_y070316009_12_ch7_K1_m3_tau3_14D_decoding_v1"
METHODS = ("peak_power", "peak_frequency", "power_frequency", "all_bands", "geometry")
DIMENSIONS = (1, 1, 2, 5, 14)
GEOMETRY_COLUMNS = ["R1", "R2", "mse", "mean_error", "frac_inside",
                    "normal_x", "normal_y", "normal_z", "u_x", "u_y", "u_z",
                    "v_x", "v_y", "v_z"]
DIRECTIONS = np.arange(1, 7)


def geometry_features(result):
    missing = np.full(14, np.nan)
    if result["status"] != "returned":
        return missing, "fit_failed: "+result.get("reason", "unknown")
    summary, fit = result["summary"], result["fit"]
    if not summary["optimizer_success"]:
        return missing, "nonconvergence"
    invalid_flags = ("nonfinite_or_zero_variance", "degenerate_radius", "collapsed_hole",
                     "invalid_inner_outer_radii", "degenerate_orientation")
    if any(flag in summary["flags"].split(";") for flag in invalid_flags):
        return missing, summary["flags"]
    values = np.r_[fit["R1"], fit["R2"], fit["R1_in"], fit["R2_in"],
                   fit["mse"], fit["mean_error"], fit["frac_inside"],
                   fit["direction"], fit["u_axis"], fit["v_axis"]]
    if not np.isfinite(values).all():
        return missing, "nonfinite_features"
    if not (0 < fit["R1_in"] < fit["R1"] and 0 < fit["R2_in"] < fit["R2"]):
        return missing, "invalid_inner_outer_radii"
    if fit["mse"] < 0 or fit["mean_error"] < 0 or not 0 <= fit["frac_inside"] <= 1:
        return missing, "invalid_quality_metric"
    frame = np.column_stack([fit["u_axis"], fit["v_axis"], fit["direction"]])
    if not np.allclose(frame.T@frame, np.eye(3), atol=1e-5):
        return missing, "invalid_orientation_frame"
    # Reuse the established canonical frame; delete only its tube/width slot.
    features = np.delete(pack_torus(fit), 2)
    if features.shape != (14,) or not np.isfinite(features).all():
        return missing, "invalid_packed_features"
    return features, ""


def feature_sets(power, frequency, bands, geometry):
    arrays = dict(peak_power=power, peak_frequency=frequency,
                  power_frequency=np.hstack([power, frequency]),
                  all_bands=bands, geometry=geometry)
    for (name, values), dimension in zip(arrays.items(), DIMENSIONS):
        if values.ndim != 2 or values.shape != (len(geometry), dimension):
            raise ValueError(f"Wrong feature dimensions: {name}: {values.shape}")
    return arrays


def decode(features, labels, folds, retain_models=False):
    labels, folds = np.asarray(labels), np.asarray(folds)
    if set(np.unique(folds)) != set(range(5)) or not np.isin(labels, DIRECTIONS).all():
        raise ValueError("Expected five saved folds and direction labels 1-6")
    predictions = np.zeros((len(labels), len(features)), dtype=np.int8)
    assigned = np.zeros(len(labels), dtype=np.int8)
    models, audit = {}, []
    for fold in range(5):
        train, test = folds != fold, folds == fold
        if set(np.unique(labels[train])) != set(DIRECTIONS):
            raise ValueError("A training fold is missing a class")
        assigned[test] += 1
        for column, (method, x) in enumerate(features.items()):
            if len(x) != len(labels) or np.isinf(x).any():
                raise ValueError(f"Invalid feature array: {method}")
            if not np.isfinite(x[train]).any(axis=0).all():
                raise ValueError(f"No usable training values: {method}, fold {fold}")
            estimator = make_estimator()
            assert list(estimator.named_steps) == ["impute", "scale", "lda"]
            estimator.fit(x[train], labels[train])
            predictions[test, column] = estimator.predict(x[test])
            if retain_models:
                models[(method, fold)] = estimator
                audit.append(dict(method=method, fold=fold, n_train=int(train.sum()),
                                  n_test=int(test.sum()), n_features=x.shape[1],
                                  imputation_medians=estimator.named_steps["impute"].statistics_,
                                  scaler_mean=estimator.named_steps["scale"].mean_,
                                  scaler_scale=estimator.named_steps["scale"].scale_,
                                  solver="lsqr", shrinkage="auto", steps=list(estimator.named_steps)))
    if not (assigned == 1).all() or not np.isin(predictions, DIRECTIONS).all():
        raise AssertionError("Each trial must receive exactly one held-out prediction per method")
    return predictions, models, audit


def score_predictions(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if len(labels) != len(predictions) or not np.isin(labels, DIRECTIONS).all() or not np.isin(predictions, DIRECTIONS).all():
        raise ValueError("Invalid direction predictions")
    counts = np.stack([np.bincount((labels-1)*6+(p-1), minlength=36).reshape(6, 6)
                       for p in predictions.T])
    tp = np.diagonal(counts, axis1=1, axis2=2)
    denom = counts.sum(axis=1)+counts.sum(axis=2)
    f1 = np.divide(2*tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    row_totals = counts.sum(axis=2, keepdims=True)
    fraction = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals > 0)
    return dict(macro_f1=f1.mean(axis=1), accuracy=tp.sum(axis=1)/len(labels),
                per_direction_f1=f1, confusion_counts=counts, confusion_fraction=fraction)


def bootstrap_predictions(labels, predictions, n_bootstrap=2000, seed=42000):
    rng = np.random.default_rng(seed)
    by_class = [np.flatnonzero(labels == d) for d in DIRECTIONS]
    indices = np.stack([np.concatenate([rng.choice(ids, len(ids), replace=True) for ids in by_class])
                        for _ in range(n_bootstrap)]).astype(np.int16)
    scores = [score_predictions(labels[idx], predictions[idx]) for idx in indices]
    return dict(indices=indices, **{name: np.stack([score[name] for score in scores])
                                   for name in ("macro_f1", "accuracy", "per_direction_f1")})


def permutation_result(index, features, labels, folds, seed=52000):
    shuffled = permute_within_folds(labels, folds, seed+index)
    predictions, _, _ = decode(features, shuffled, folds)
    scores = score_predictions(shuffled, predictions)
    return dict(index=index, labels=shuffled.astype(np.int8), predictions=predictions,
                macro_f1=scores["macro_f1"], accuracy=scores["accuracy"])
