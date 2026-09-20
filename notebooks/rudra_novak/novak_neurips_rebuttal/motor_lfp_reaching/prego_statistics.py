"""Summaries computed from frozen held-out predictions, never from fitted scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def prediction_metrics(labels, predicted):
    classes = f1_score(labels, predicted, labels=np.arange(1, 7), average=None, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=np.arange(1, 7)).astype(float)
    if np.any(matrix.sum(axis=1) == 0):
        raise ValueError("Missing true direction")
    matrix /= matrix.sum(axis=1, keepdims=True)
    return float(classes.mean()), classes, matrix


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def cluster_interval(differences, seed=42, draws=10000):
    grouped = differences.groupby("day")["difference"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), (draws, len(grouped)))
    boot = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped["count"].to_numpy()[indices].sum(axis=1)
    return (float(differences.difference.mean()), *np.quantile(boot, [.025, .975]).tolist())


def select_example(recordings):
    valid = recordings[np.isfinite(recordings.fit_error)].copy()
    valid["distance"] = abs(valid.fit_error - valid.fit_error.median())
    return valid.sort_values(["distance", "recording"]).iloc[0].recording
