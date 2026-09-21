"""Leakage-controlled, single-channel pre-GO feature extraction and decoding."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime

import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from motor_lfp_utils import BANDS, fit_elliptical_torus_3d, lag_embed
from select_trace_embedding_parameters import (
    adjust_embedding_for_epoch_length, bootstrap_peak_support, compute_log_psd,
    detect_peaks, preprocess_segments, remove_aperiodic_background,
    select_embedding_dim, select_tau,
)

DIRECTIONS = np.arange(1, 7)
TORUS_COLUMNS = ["R1", "R2", "r", "mse", "mean_error", "frac_inside",
                 "direction_x", "direction_y", "direction_z", "u_x", "u_y", "u_z",
                 "v_x", "v_y", "v_z"]
MAIN_METHODS = ["peak_power", "peak_frequency", "power_frequency", "all_bands", "torus"]
METHODS = MAIN_METHODS + ["torus_power_frequency", "torus_all_bands", "torus_pca"]


def stable_seed(name, seed=42):
    return (int(hashlib.sha256(str(name).encode()).hexdigest()[:8], 16) + seed) % (2**32 - 1)


def extract_prego(lfp, fs=1000, go_sample=4500):
    length = int(round(.5 * fs))
    if fs != 1000 or go_sample < length or go_sample > lfp.shape[1]:
        raise ValueError("Expected 1 kHz data with a complete 500-ms pre-GO epoch")
    segments = np.asarray(lfp[:, go_sample-length:go_sample], dtype=float)
    good = np.isfinite(segments).all(axis=1) & (np.ptp(segments, axis=1) > 1e-12)
    return segments[good], np.flatnonzero(good)


def spectral_features(segments, monkey):
    if monkey not in ("M", "T"):
        raise ValueError(monkey)
    x = signal.detrend(np.asarray(segments, dtype=float), axis=1, type="linear")
    freq, psd = signal.welch(x, fs=1000, window="hamming", nperseg=300,
                             noverlap=200, nfft=10000, detrend=False, axis=1)
    bounds = [(12, 25), (25, 40)] if monkey == "M" else [(12, 40)]
    power, frequency = [], []
    for low, high in bounds:
        mask = (freq >= low) & (freq <= high)
        idx = np.argmax(psd[:, mask], axis=1)
        power.append(np.log10(np.maximum(psd[:, mask][np.arange(len(x)), idx], 1e-30)))
        frequency.append(freq[mask][idx])
    # Zero padding permits band integration at exact boundaries; it adds no resolution.
    f, density = signal.welch(x, fs=1000, window="hann", nperseg=500,
                              noverlap=250, nfft=10000, detrend=False, axis=1)
    band_values = []
    for low, high in BANDS.values():
        mask = (f >= low) & (f <= high)
        band_values.append(np.log10(np.maximum(np.trapezoid(density[:, mask], f[mask], axis=1), 1e-30)))
    return np.array(power).T, np.array(frequency).T, np.array(band_values).T


def balanced_indices(y, seed=42, count=None):
    available = [np.flatnonzero(y == label) for label in DIRECTIONS]
    smallest = min(map(len, available))
    n = smallest if count is None else count
    if n < 15 or n > smallest:
        raise ValueError("Require at least 15 valid trials in every direction")
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(idx)[:n] for idx in available])


def fold_assignments(y, seed=42):
    folds = np.empty(len(y), dtype=int)
    for fold, (_, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)):
        folds[test] = fold
    return folds


def feature_sets(power, frequency, bands, torus):
    joint = np.hstack([power, frequency])
    return dict(peak_power=power, peak_frequency=frequency, power_frequency=joint,
                all_bands=bands, torus=torus, torus_pca=torus,
                torus_power_frequency=np.hstack([torus, joint]),
                torus_all_bands=np.hstack([torus, bands]))


def make_estimator(pca_dim=None):
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, svd_solver="full")))
    return Pipeline(steps + [("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])


def recording_day(session):
    match = re.match(r"[oy](\d{6})\d", session)
    if match is None:
        raise ValueError(f"Cannot infer recording day from {session}")
    return datetime.strptime(match.group(1), "%y%m%d").strftime("%Y-%m-%d")


def positive_axis(axis):
    a = np.asarray(axis, dtype=float).copy()
    a /= np.linalg.norm(a)
    return a if a[np.argmax(np.abs(a))] >= 0 else -a


def pack_torus(fit):
    r1, r2 = fit["R1"], fit["R2"]
    u = fit["u_axis"] if r1 >= r2 else fit["v_axis"]
    n = positive_axis(fit["direction"])
    u = positive_axis(u - np.dot(u, n) * n)
    v = np.cross(n, u)
    return np.r_[max(r1, r2), min(r1, r2), fit["minor_radius"], fit["mse"],
                 fit["mean_error"], fit["frac_inside"], n, u, v]


def learn_embedding(training_segments, seed=42, bootstraps=120):
    processed = preprocess_segments(training_segments)
    tau, method, _, _, _ = select_tau(processed, 100, seed)
    freqs, log_psds = compute_log_psd(training_segments, 55)
    residuals = remove_aperiodic_background(freqs, log_psds)
    candidates, _, _ = detect_peaks(freqs, np.median(residuals, axis=0))
    support = bootstrap_peak_support(freqs, residuals, candidates, seed, bootstraps)
    robust = candidates[support >= .5]
    _, dim, _ = select_embedding_dim(len(robust))
    dim, tau, adjustment = adjust_embedding_for_epoch_length(dim, tau, processed.shape[1], 300)
    mean, components = np.zeros(dim), np.eye(dim)[:3]
    if dim > 3:
        cloud = np.vstack([lag_embed(x, dim=dim, tau=tau)[::4] for x in processed])
        projection = PCA(3, svd_solver="full").fit(cloud)
        mean = projection.mean_
        components = np.array([positive_axis(row) for row in projection.components_])
    return dict(dimension=int(dim), tau=int(tau), tau_method=method,
                n_training_trials=len(training_segments), robust_peaks=robust.tolist(),
                adjustment=adjustment, center=mean.tolist(), components=components.tolist())


def geometry_coordinates(segment, embedding):
    processed = preprocess_segments(np.asarray(segment)[None])[0]
    cloud = lag_embed(processed, dim=embedding["dimension"], tau=embedding["tau"])
    return (cloud - np.array(embedding["center"])) @ np.array(embedding["components"]).T


def fit_geometry(segment, embedding, seed=42):
    points = geometry_coordinates(segment, embedding)
    if len(points) > 300:
        points = points[np.sort(np.random.default_rng(seed).choice(len(points), 300, replace=False))]
    try:
        fitted = fit_elliptical_torus_3d(points, max_nfev=1200)
        features = pack_torus(fitted)
        valid = bool(fitted["success"] and np.isfinite(features).all())
        return (features if valid else np.full(15, np.nan)), dict(
            success=valid, nfev=fitted["nfev"], cost=fitted["cost"],
            mse=fitted["mse"], reason="" if valid else "optimizer_not_converged")
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        return np.full(15, np.nan), dict(success=False, nfev=0, cost=np.nan,
                                       mse=np.nan, reason=str(error))


def permute_within_folds(y, folds, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.array(y, copy=True)
    for fold in np.unique(folds):
        mask = folds == fold
        shuffled[mask] = rng.permutation(y[mask])
    return shuffled


def heldout_predictions(power, frequency, bands, torus_by_fold, labels, folds, methods=METHODS):
    predictions = {method: np.zeros(len(labels), dtype=int) for method in methods}
    for fold in range(5):
        train, test = folds != fold, folds == fold
        geometry = torus_by_fold[fold]
        if not np.isfinite(geometry[train]).any(axis=0).all():
            raise ValueError(f"Fold {fold} has no usable training geometry")
        features = feature_sets(power, frequency, bands, geometry)
        for method in methods:
            estimator = make_estimator(2*power.shape[1] if method == "torus_pca" else None)
            predictions[method][test] = estimator.fit(features[method][train], labels[train]).predict(features[method][test])
    return predictions
