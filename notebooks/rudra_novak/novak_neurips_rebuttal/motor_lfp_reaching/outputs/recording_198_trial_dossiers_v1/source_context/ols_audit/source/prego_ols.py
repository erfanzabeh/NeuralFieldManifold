"""Trial-local, nested-validation OLS diagnostics. No geometric model or decoder."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ORIGINS = np.arange(200, 490)
HORIZONS = (1, 10)


def default_config():
    return dict(version=1, fs=1000, epoch_samples=500, origins=[200, 489],
                orders=list(range(1, 21)), dimensions=list(range(2, 10)),
                delays=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
                max_span=200, inner_folds=3, outer_folds=5, seed=20260921,
                horizons=[1, 10], preprocessing="per-trial detrend, median/MAD, 2-55 Hz zero-phase",
                interpretation="offline prediction of preprocessed signals",
                selection="10ms inner-fold NMSE; one-SE; simpler dimension/order first")


def candidates(config, family):
    if family == "ar":
        return [dict(p=p, m=0, tau=1) for p in config["orders"]]
    return [dict(p=0, m=m, tau=t) for m in config["dimensions"]
            for t in config["delays"] if (m - 1) * t <= config["max_span"]]


def design(x, delays, horizons):
    x = np.asarray(x, dtype=float)
    delays, horizons = np.asarray(delays), np.asarray(horizons)
    if (x.ndim != 2 or x.shape[1] != 500 or np.any(delays < 0)
            or np.max(delays) > ORIGINS[0] or np.min(horizons) < 1
            or ORIGINS[-1] + np.max(horizons) >= x.shape[1]):
        raise ValueError("Predictors and targets must stay inside each 500-sample epoch")
    return x[:, ORIGINS[:, None] - delays], x[:, ORIGINS[:, None] + horizons]


def fit_ols(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, x.shape[-1])
    y = np.asarray(y, dtype=float).reshape(-1, y.shape[-1])
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    # Standardization improves numerical scaling but is invertible, not regularization.
    a = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    weights, _, rank, singular = np.linalg.lstsq(a, y, rcond=None)
    slopes = weights[1:] / scale[:, None]
    coefficients = np.vstack((weights[0] - mean @ slopes, slopes))
    raw = np.column_stack((np.ones(len(x)), x))
    raw_singular = np.linalg.svd(raw, compute_uv=False)
    return dict(coefficients=coefficients, rank=int(rank), n_columns=a.shape[1],
                condition=float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf,
                condition_original=float(raw_singular[0] / raw_singular[-1]) if raw_singular[-1] > 0 else np.inf,
                singular_values=singular, column_mean=mean, column_scale=scale,
                status="ok" if rank == a.shape[1] else "rank_deficient")


def predict(model, x):
    coefficients = np.asarray(model["coefficients"])
    return coefficients[0] + x @ coefficients[1:]


def recursive_prediction(coefficients, history, horizon):
    c = np.asarray(coefficients, dtype=float)
    transform, bias = np.eye(len(c) - 1), np.zeros(len(c) - 1)
    # Compose the affine AR recurrence; every intermediate value is predicted.
    for _ in range(horizon):
        row, intercept = c[1:] @ transform, c[0] + c[1:] @ bias
        transform = np.vstack((row, transform[:-1]))
        bias = np.r_[intercept, bias[:-1]]
    return history @ transform[0] + bias[0]


def trial_metrics(truth, predicted, baseline):
    truth, predicted, baseline = map(lambda a: np.asarray(a, dtype=float), (truth, predicted, baseline))
    centered = truth - truth.mean(axis=1, keepdims=True)
    pred_centered = predicted - predicted.mean(axis=1, keepdims=True)
    base_centered = baseline - baseline.mean(axis=1, keepdims=True)
    tss = np.sum(centered**2, axis=1)
    mse = np.mean((truth - predicted)**2, axis=1)
    base_mse = np.mean((truth - baseline)**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        nmse = np.where(tss > 0, mse * truth.shape[1] / tss, np.nan)
        base_nmse = np.where(tss > 0, base_mse * truth.shape[1] / tss, np.nan)
        correlation = np.sum(centered * pred_centered, axis=1) / np.sqrt(tss * np.sum(pred_centered**2, axis=1))
        base_correlation = np.sum(centered * base_centered, axis=1) / np.sqrt(tss * np.sum(base_centered**2, axis=1))
        skill = np.where(base_nmse > 0, 1 - nmse / base_nmse, np.nan)
    return dict(mse=mse, nmse=nmse, r2=1-nmse, pearson=correlation,
                nmse_persistence=base_nmse, r2_persistence=1-base_nmse,
                pearson_persistence=base_correlation,
                nmse_improvement=base_nmse-nmse, persistence_skill=skill)


def select_candidate(rows, family):
    good = rows[rows.valid & np.isfinite(rows.mean_nmse) & np.isfinite(rows.se_nmse)].copy()
    if good.empty:
        return dict(status="no_valid_candidate")
    best = good.sort_values(["mean_nmse", "p", "m", "tau"], kind="stable").iloc[0]
    threshold = float(best.mean_nmse + best.se_nmse)
    near = good[good.mean_nmse <= threshold]
    ordering = ["p", "mean_nmse", "tau"] if family == "ar" else ["m", "mean_nmse", "tau"]
    chosen = near.sort_values(ordering, kind="stable").iloc[0]
    return dict(status="ok", p=int(chosen.p), m=int(chosen.m), tau=int(chosen.tau),
                mean_nmse=float(chosen.mean_nmse), se_nmse=float(chosen.se_nmse),
                best_mean_nmse=float(best.mean_nmse), one_se_threshold=threshold)


def ar_poles(coefficients):
    roots = np.roots(np.r_[1., -np.asarray(coefficients)[1:]])
    return pd.DataFrame(dict(real=roots.real, imaginary=roots.imag,
                             magnitude=np.abs(roots), frequency_hz=np.angle(roots)*1000/(2*np.pi)))


def _delays(candidate, family):
    return np.arange(candidate["p"]) if family == "ar" else np.arange(candidate["m"]) * candidate["tau"]


def _predictions(model, predictors, family):
    if family == "ar":
        return np.stack([recursive_prediction(model["coefficients"][:, 0], predictors, h)
                         for h in HORIZONS], axis=-1)
    return predict(model, predictors)


def audit_fold(x, labels, folds, outer_fold, config, seed=0):
    train, test = np.flatnonzero(folds != outer_fold), np.flatnonzero(folds == outer_fold)
    splitter = StratifiedKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed+outer_fold)
    splits = [dict(train=train[a], validation=train[b]) for a, b in splitter.split(train, labels[train])]
    inner_rows, validation_rows, selection, models = [], [], {}, {}
    prediction, metric_rows, poles = {}, [], []
    _, targets = design(x, [0], HORIZONS)
    baseline = x[:, ORIGINS]
    for family in ("ar", "embedding"):
        for candidate in candidates(config, family):
            predictors, _ = design(x, _delays(candidate, family), HORIZONS)
            candidate_rows = []
            for inner_fold, split in enumerate(splits):
                a, b = split["train"], split["validation"]
                model = fit_ols(predictors[a], targets[a, :, :1] if family == "ar" else targets[a])
                predicted = _predictions(model, predictors[b], family)
                for j, horizon in enumerate(HORIZONS):
                    metrics = trial_metrics(targets[b, :, j], predicted[:, :, j], baseline[b])
                    means = {key: float(np.mean(value)) for key, value in metrics.items()}
                    row = dict(family=family, **candidate, inner_fold=inner_fold, horizon_ms=horizon,
                               status=model["status"], rank=model["rank"], n_columns=model["n_columns"],
                               condition=model["condition"], condition_original=model["condition_original"],
                               n_trials=len(b), nmse_undefined=int((~np.isfinite(metrics["nmse"])).sum()),
                               pearson_undefined=int((~np.isfinite(metrics["pearson"])).sum()),
                               coefficients=json.dumps(model["coefficients"].tolist()), **means)
                    candidate_rows.append(row)
                    inner_rows.append(row)
            for horizon in HORIZONS:
                rows = [r for r in candidate_rows if r["horizon_ms"] == horizon]
                errors = np.array([r["nmse"] for r in rows])
                validation_rows.append(dict(family=family, **candidate, horizon_ms=horizon,
                    mean_nmse=float(errors.mean()), se_nmse=float(errors.std(ddof=1)/np.sqrt(len(errors))),
                    valid=all(r["status"] == "ok" and np.isfinite(r["nmse"]) for r in rows),
                    mean_persistence=float(np.mean([r["nmse_persistence"] for r in rows])),
                    max_condition=float(max(r["condition"] for r in rows))))
        frame = pd.DataFrame(validation_rows)
        chosen = select_candidate(frame[(frame.family == family) & (frame.horizon_ms == 10)], family)
        chosen["boundary"] = bool(chosen.get("p") == max(config["orders"])) if family == "ar" else bool(
            chosen.get("m") == max(config["dimensions"]) or chosen.get("tau") in (min(config["delays"]), max(config["delays"])))
        selection[family] = chosen
        if chosen["status"] != "ok":
            prediction[family] = np.full_like(targets[test], np.nan)
            continue
        predictors, _ = design(x, _delays(chosen, family), HORIZONS)
        model = fit_ols(predictors[train], targets[train, :, :1] if family == "ar" else targets[train])
        models[family] = model
        predicted = _predictions(model, predictors[test], family)
        prediction[family] = predicted
        if family == "ar":
            poles = ar_poles(model["coefficients"][:, 0]).to_dict("records")
        for j, horizon in enumerate(HORIZONS):
            metrics = trial_metrics(targets[test, :, j], predicted[:, :, j], baseline[test])
            for k, index in enumerate(test):
                metric_rows.append(dict(family=family, horizon_ms=horizon, trial_index=int(index),
                    status=model["status"], **{key: value[k] for key, value in metrics.items()}))
    return dict(validation=pd.DataFrame(validation_rows), inner_scores=pd.DataFrame(inner_rows),
                selection=selection, models=models, poles=poles, inner_splits=splits,
                metrics=pd.DataFrame(metric_rows), test_indices=test,
                predictions=dict(truth=targets[test], persistence=baseline[test], **prediction))
