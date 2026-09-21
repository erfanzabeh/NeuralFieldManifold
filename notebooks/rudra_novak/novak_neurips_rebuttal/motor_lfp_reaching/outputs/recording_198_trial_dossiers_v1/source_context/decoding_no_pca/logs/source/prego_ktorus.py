"""PSD-selected, full-delay-coordinate K-mode geometry. No PCA or PINN."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from motor_lfp_utils import lag_embed
from select_trace_embedding_parameters import (
    average_mutual_information, bootstrap_peak_support, compute_log_psd,
    detect_peaks, preprocess_segments, remove_aperiodic_background,
)

METHODS = ["peak_power", "peak_frequency", "power_frequency", "all_bands",
           "ktorus", "ktorus_power_frequency", "ktorus_all_bands"]
MAIN_METHODS = METHODS[:5]


def delay_map(frequencies, dimension, tau):
    phase = 2*np.pi*np.arange(dimension)[:, None]*tau*np.asarray(frequencies)[None, :]/1000
    return np.stack([np.cos(phase), np.sin(phase)], axis=-1).reshape(dimension, -1)


def harmonic_flags(frequencies):
    flags = []
    for i, low in enumerate(frequencies):
        for high in frequencies[i+1:]:
            ratio = int(round(high/low))
            if ratio >= 2 and abs(high-ratio*low) <= 2:
                flags.append(dict(lower_hz=float(low), upper_hz=float(high), multiple=ratio))
    return flags


def choose_embedding(frequencies, lags, ami, length, min_points=300, max_condition=100.):
    frequencies = np.asarray(frequencies, dtype=float)
    k, dimension = len(frequencies), 2*len(frequencies)+1
    result = dict(K=k, dimension=dimension, tau=None, status="unresolved_embedding",
                  frequencies=frequencies.tolist(), candidates=[])
    if k == 0:
        result["status"] = "no_supported_peaks"
        return result
    if not np.isfinite(ami).all():
        result["reason"] = "nonfinite_ami"
        return result
    smooth = gaussian_filter1d(np.asarray(ami, dtype=float), 1.)
    minima = np.flatnonzero((smooth[1:-1] <= smooth[:-2]) & (smooth[1:-1] <= smooth[2:]))+1
    for index in minima:
        tau = int(lags[index])
        n_points = length-(dimension-1)*tau
        basis = delay_map(frequencies, dimension, tau)
        rank = int(np.linalg.matrix_rank(basis))
        condition = float(np.linalg.cond(basis))
        reason = ("insufficient_points" if n_points < min_points else
                  "rank_deficient" if rank != 2*k else
                  "ill_conditioned" if condition > max_condition else "accepted")
        result["candidates"].append(dict(tau=tau, n_points=n_points, rank=rank,
                                          condition=condition, reason=reason))
        if reason == "accepted" and result["tau"] is None:
            result.update(tau=tau, status="ok", condition=condition, n_points=n_points)
    return result


def learn_geometry(training_segments, seed=42, bootstraps=120):
    x = np.asarray(training_segments)
    if x.ndim != 2 or x.shape[1] != 500 or not np.isfinite(x).all():
        raise ValueError("Expected finite training-only 500-sample epochs")
    freq, log_psd = compute_log_psd(x, 55.)
    residual = remove_aperiodic_background(freq, log_psd)
    peaks, props, smooth_psd = detect_peaks(freq, np.median(residual, axis=0))
    support = bootstrap_peak_support(freq, residual, peaks, seed, bootstraps)
    selected = peaks[support >= .5]
    processed = preprocess_segments(x)
    lags = np.arange(1, 101)
    ami = average_mutual_information(processed, lags, 48, 80000, seed)
    result = choose_embedding(selected, lags, ami, x.shape[1])
    result.update(n_training_trials=len(x), seed=int(seed), bootstraps=bootstraps,
                  harmonic_ambiguities=harmonic_flags(selected),
                  psd_frequencies=freq.tolist(), psd_log_median=np.median(log_psd, axis=0).tolist(),
                  residual_median=np.median(residual, axis=0).tolist(), psd_smoothed=smooth_psd.tolist(),
                  peak_candidates=peaks.tolist(), peak_support=support.tolist(),
                  peak_prominence=props["prominences"].tolist(),
                  ami_lags=lags.tolist(), ami=ami.tolist(),
                  feature_names=feature_names(result["K"], result["dimension"]))
    return result


def phase_design(times, frequencies):
    phase = 2*np.pi*np.asarray(times)[:, None]*np.asarray(frequencies)[None, :]
    modes = np.stack([np.cos(phase), np.sin(phase)], axis=-1).reshape(len(times), -1)
    return np.column_stack([np.ones(len(times)), modes])


def frequency_bounds(frequencies):
    frequency = np.asarray(frequencies, dtype=float)
    lower, upper = np.maximum(2., frequency-2.), np.minimum(55., frequency+2.)
    if len(frequency) > 1:
        middle = (frequency[:-1]+frequency[1:])/2
        lower[1:] = np.maximum(lower[1:], middle+1e-6)
        upper[:-1] = np.minimum(upper[:-1], middle-1e-6)
    return lower, upper


def fit_cloud(points, times, frequencies, max_nfev=120):
    points, times = np.asarray(points, dtype=float), np.asarray(times, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    if (points.ndim != 2 or len(times) != len(points) or len(frequency) == 0
            or len(points) <= 1+2*len(frequency) or not np.isfinite(points).all()
            or np.any(np.diff(frequency) <= 0)):
        raise ValueError("Invalid full-dimensional cloud or ordered nonempty mode frequencies")

    def solve(freq):
        design = phase_design(times, freq)
        coefficients, _, rank, _ = np.linalg.lstsq(design, points, rcond=None)
        return coefficients, design @ coefficients, rank, design

    # Variable projection eliminates linear coefficients without projecting data
    # into a reduced observation space. Every original lag contributes residuals.
    def residual(freq):
        return (solve(freq)[1]-points).ravel()

    lower, upper = frequency_bounds(frequency)
    fit = least_squares(residual, np.clip(frequency, lower+1e-8, upper-1e-8),
                        bounds=(lower, upper), max_nfev=max_nfev,
                        ftol=1e-7, xtol=1e-7, gtol=1e-7)
    coefficients, predicted, rank, design = solve(fit.x)
    errors = points-predicted
    denominator = np.sum((points-points.mean(axis=0))**2)
    normalized = float(np.sum(errors**2)/max(denominator, 1e-30))
    shape_rank = int(np.linalg.matrix_rank(coefficients[1:].T))
    largest_gaps = []
    for f in fit.x:
        phase = np.sort(np.mod(f*times, 1.))
        largest_gaps.append(float(np.diff(np.r_[phase, phase[0]+1]).max()))
    return dict(success=bool(fit.success and rank == 1+2*len(frequency)
                             and np.isfinite(coefficients).all() and denominator > 1e-20),
                coefficients=coefficients, prediction=predicted, frequencies=fit.x,
                normalized_error=normalized, rmse_by_coordinate=np.sqrt(np.mean(errors**2, axis=0)),
                nfev=int(fit.nfev), reason="" if fit.success else str(fit.message),
                temporal_design_rank=int(rank), temporal_condition=float(np.linalg.cond(design)),
                shape_rank=shape_rank, shape_condition=float(np.linalg.cond(coefficients[1:].T)),
                frequency_at_bound=bool(np.any(np.minimum(fit.x-lower, upper-fit.x) < .01)),
                phase_largest_gaps=largest_gaps)


def mode_shapes(coefficients):
    blocks = np.asarray(coefficients)[1:].reshape(-1, 2, coefficients.shape[1])
    return np.einsum("kri,krj->kij", blocks, blocks)


def feature_names(k, dimension):
    names = [f"mode_{mode+1}_G_{i}_{j}" for mode in range(k)
             for i, j in zip(*np.triu_indices(dimension))]
    return names + [f"residual_rmse_lag_{i}" for i in range(dimension)] + ["normalized_error"]


def geometry_features(result, points):
    dimension = points.shape[1]
    shapes = mode_shapes(result["coefficients"])
    i, j = np.triu_indices(dimension)
    return np.r_[shapes[:, i, j].ravel(), result["rmse_by_coordinate"], result["normalized_error"]]


def trial_cloud(segment, embedding):
    x = np.asarray(segment, dtype=float)
    if x.ndim != 1 or len(x) != 500 or not np.isfinite(x).all() or np.ptp(x) < 1e-12:
        raise ValueError("Invalid trial")
    processed = preprocess_segments(x[None])[0]
    points = lag_embed(processed, embedding["dimension"], embedding["tau"])
    times = (np.arange(len(points))+(embedding["dimension"]-1)*embedding["tau"])/1000
    return points, times


def fit_trial(segment, embedding):
    width = len(feature_names(embedding["K"], embedding["dimension"]))
    try:
        points, times = trial_cloud(segment, embedding)
        result = fit_cloud(points, times, embedding["frequencies"])
        features = geometry_features(result, points)
        result.pop("prediction")
        return features if result["success"] else np.full(width, np.nan), result
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return np.full(width, np.nan), dict(success=False, reason=str(error))


def estimator():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def matrices(power, frequency, bands, geometry):
    joint = np.hstack([power, frequency])
    return dict(peak_power=power, peak_frequency=frequency, power_frequency=joint,
                all_bands=bands, ktorus=geometry,
                ktorus_power_frequency=np.hstack([geometry, joint]),
                ktorus_all_bands=np.hstack([geometry, bands]))


def decode(power, frequency, bands, geometry_by_fold, labels, folds, methods=METHODS):
    predictions = {method: np.zeros(len(labels), dtype=int) for method in methods}
    for fold in range(5):
        train, test = folds != fold, folds == fold
        geom = geometry_by_fold[fold]
        if any("ktorus" in method for method in methods) and not np.isfinite(geom[train]).any(axis=0).all():
            raise ValueError(f"Fold {fold}: no usable training geometry")
        features = matrices(power, frequency, bands, geom)
        for method in methods:
            predictions[method][test] = estimator().fit(features[method][train], labels[train]).predict(features[method][test])
    return predictions
