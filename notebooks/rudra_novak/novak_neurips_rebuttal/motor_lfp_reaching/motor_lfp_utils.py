#!/usr/bin/env python
"""Utilities for macaque motor-cortex LFP reaching analyses."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import io, optimize, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


UNIT_DIR = Path(__file__).resolve().parent
RAW_DIR = UNIT_DIR / "raw_data"
CONVERTED_DIR = UNIT_DIR / "lfp_converted_data"
CACHE_DIR = UNIT_DIR / "cache"
PLOT_DIR = UNIT_DIR / "plots"
TABLE_DIR = UNIT_DIR / "tables"

FS = 1000
GO_SAMPLE = 4500
RANDOM_SEED = 42
MIN_CLASS_COUNT = 5
N_SPLITS = 5

DIRECTION_LABELS = [1, 2, 3, 4, 5, 6]
DELAY_LABELS = ["short", "long"]

BANDS = {
    "delta": (2.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 55.0),
}

FEATURE_ORDER = [*BANDS.keys(), "average_psd", "all_band_power", "torus_geometry_15"]
FEATURE_LABELS = {
    "delta": "Delta\n(2-4 Hz)",
    "theta": "Theta\n(4-8 Hz)",
    "alpha": "Alpha\n(8-13 Hz)",
    "beta": "Beta\n(13-30 Hz)",
    "low_gamma": "Low gamma\n(30-55 Hz)",
    "average_psd": "Average\nPSD",
    "all_band_power": "All band\npower",
    "torus_geometry_15": "Torus\nfeatures",
}

TORUS_PARAM_DEFAULT_SOURCE = "cli_default"

EPOCHS = {
    "pre_go": ("go", -1.0, 0.0),
    "peri_go": ("go", -0.25, 0.75),
    "movement": ("movement", -0.25, 0.75),
    "post_go": ("go", 0.0, 1.0),
}


def ensure_dirs() -> None:
    for path in (RAW_DIR, CONVERTED_DIR, CACHE_DIR, PLOT_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def slugify(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text or "unknown"


def mat_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return default


def matlab_scalar(value: Any) -> Any:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        return matlab_scalar(arr.item())
    if arr.size == 0:
        return None
    if arr.size == 1:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return item
    return value


def matlab_string(value: Any) -> str:
    scalar = matlab_scalar(value)
    if scalar is None:
        return ""
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    arr = np.asarray(scalar)
    if arr.dtype.kind in {"U", "S"}:
        return "".join(arr.ravel().astype(str)).strip()
    return str(scalar)


def matlab_int(value: Any, default: int = -1) -> int:
    scalar = matlab_scalar(value)
    try:
        if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
            return default
        return int(scalar)
    except (TypeError, ValueError):
        return default


def condition_cells(value: Any) -> list[Any]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object)
    if arr.dtype == object:
        return [x for x in arr.ravel()]
    return [value]


def cell_value(value: Any, cond_idx: int, trial_idx: int | None = None) -> Any:
    cells = condition_cells(value)
    if not cells:
        return None
    cell = cells[min(cond_idx, len(cells) - 1)]
    if trial_idx is None:
        return cell

    arr = np.asarray(cell, dtype=object if np.asarray(cell).dtype == object else None)
    if arr.dtype == object:
        flat = arr.ravel()
        if flat.size == 0:
            return None
        return flat[min(trial_idx, flat.size - 1)]

    arr = np.asarray(cell)
    if arr.ndim == 0:
        return arr.item()
    if arr.ndim == 1:
        return arr[min(trial_idx, arr.shape[0] - 1)]
    if arr.shape[0] > trial_idx:
        return arr[trial_idx]
    if arr.shape[1] > trial_idx:
        return arr[:, trial_idx]
    return None


def as_trial_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[1] >= GO_SAMPLE + 1:
        return arr.astype(np.float32, copy=False)
    if arr.shape[0] >= GO_SAMPLE + 1:
        return arr.T.astype(np.float32, copy=False)
    if arr.shape[0] <= arr.shape[1]:
        return arr.astype(np.float32, copy=False)
    return arr.T.astype(np.float32, copy=False)


def numeric_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    arr = np.asarray(value)
    if arr.dtype == object:
        out = []
        for item in arr.ravel():
            out.extend(numeric_vector(item).tolist())
        return np.asarray(out, dtype=float)
    try:
        return np.asarray(arr, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def parse_trial_labels(
    codes: np.ndarray,
    cond_idx: int,
    n_conditions: int,
) -> tuple[int, str, int, int]:
    codes_int = np.asarray(codes, dtype=float).ravel()
    codes_int = codes_int[np.isfinite(codes_int)].astype(int)

    condition_candidates = codes_int[(codes_int >= 1) & (codes_int <= 12)]
    condition_code = int(condition_candidates[0]) if condition_candidates.size else cond_idx + 1

    direction_candidates = codes_int[(codes_int >= 101) & (codes_int <= 106)]
    direction = int(direction_candidates[0] - 100) if direction_candidates.size else -1

    trial_type_candidates = codes_int[(codes_int == 91) | (codes_int == 92)]
    trial_type_code = int(trial_type_candidates[0]) if trial_type_candidates.size else -1
    delay_label = "short" if trial_type_code == 91 else "long" if trial_type_code == 92 else "unknown"

    if direction < 0 and n_conditions == 12:
        direction = ((condition_code - 1) % 6) + 1
    if delay_label == "unknown" and n_conditions == 12:
        delay_label = "short" if condition_code <= 6 else "long"
        trial_type_code = 91 if delay_label == "short" else 92

    return direction, delay_label, trial_type_code, condition_code


def load_mat_entries(mat_path: Path) -> tuple[str, list[Any]]:
    mat_path = Path(mat_path)
    animal = "T" if "T" in mat_path.stem else "M" if "M" in mat_path.stem else mat_path.stem
    data = io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    candidates = [(key, value) for key, value in data.items() if not key.startswith("__")]
    if not candidates:
        raise ValueError(f"No MATLAB data variables found in {mat_path}")

    _name, value = max(candidates, key=lambda kv: np.asarray(kv[1], dtype=object).size)
    arr = np.asarray(value, dtype=object)
    entries = [item for item in arr.ravel()]
    return animal, entries


def detrend_zscore(x: np.ndarray) -> np.ndarray:
    y = signal.detrend(np.asarray(x, dtype=float), type="linear")
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    scale = 1.4826 * mad if mad > 1e-12 else np.std(y)
    if scale <= 1e-12:
        return y * 0.0
    return (y - med) / scale


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def compute_band_power(segments: np.ndarray, fs: int = FS) -> np.ndarray:
    out = np.zeros((len(segments), len(BANDS)), dtype=float)
    for i, segment in enumerate(segments):
        x = detrend_zscore(segment)
        nperseg = min(1024, len(x))
        noverlap = min(nperseg // 2, max(0, nperseg - 1))
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        df = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 1.0
        for j, (_name, (low, high)) in enumerate(BANDS.items()):
            mask = (freqs >= low) & (freqs <= high)
            if not mask.any():
                out[i, j] = np.nan
            elif mask.sum() == 1:
                out[i, j] = float(psd[mask][0] * df)
            else:
                out[i, j] = float(np.trapezoid(psd[mask], freqs[mask]))
    return np.log10(out + 1e-12)


def compute_average_psd(segments: np.ndarray, fs: int = FS, low: float = 2.0, high: float = 55.0) -> np.ndarray:
    out = np.zeros((len(segments), 1), dtype=float)
    for i, segment in enumerate(segments):
        x = detrend_zscore(segment)
        nperseg = min(1024, len(x))
        noverlap = min(nperseg // 2, max(0, nperseg - 1))
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mask = (freqs >= low) & (freqs <= high)
        out[i, 0] = float(np.mean(psd[mask])) if mask.any() else np.nan
    return np.log10(out + 1e-12)


def load_torus_param_table(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None or str(path).strip() == "":
        return {}
    table_path = Path(path)
    df = pd.read_csv(table_path)
    required = {"lfp_uid", "torus_tau", "torus_embedding_dim"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{table_path} is missing required columns: {', '.join(missing)}")
    if df["lfp_uid"].duplicated().any():
        duplicated = ", ".join(df.loc[df["lfp_uid"].duplicated(), "lfp_uid"].astype(str).head(5))
        raise ValueError(f"{table_path} has duplicate lfp_uid rows, including {duplicated}")

    out: dict[str, dict[str, object]] = {}
    for row in df.to_dict("records"):
        tau = int(row["torus_tau"])
        dim = int(row["torus_embedding_dim"])
        if tau < 1:
            raise ValueError(f"Invalid tau={tau} for {row['lfp_uid']}")
        if dim < 2:
            raise ValueError(f"Invalid embedding dimension={dim} for {row['lfp_uid']}")
        out[str(row["lfp_uid"])] = {
            **row,
            "torus_tau": tau,
            "torus_embedding_dim": dim,
            "torus_param_source": str(row.get("torus_param_source", table_path.name)),
            "torus_param_id": str(row.get("torus_param_id", f"tau{tau}_embed{dim}")),
        }
    return out


def resolve_torus_params(
    lfp_uid: str,
    param_table: dict[str, dict[str, object]] | None,
    default_tau: int,
    default_embedding_dim: int,
) -> dict[str, object]:
    if param_table and lfp_uid in param_table:
        return param_table[lfp_uid]
    tau = int(default_tau)
    dim = int(default_embedding_dim)
    return {
        "lfp_uid": lfp_uid,
        "torus_tau": tau,
        "torus_embedding_dim": dim,
        "torus_param_source": TORUS_PARAM_DEFAULT_SOURCE,
        "torus_param_id": f"tau{tau}_embed{dim}",
    }


def lag_embed(x: np.ndarray, dim: int = 3, tau: int = 20) -> np.ndarray:
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        return np.empty((0, dim), dtype=float)
    rows = np.arange(n)[:, None]
    cols = (dim - 1 - np.arange(dim)) * tau
    return np.asarray(x, dtype=float)[rows + cols]


def torus_fit_coordinates(points: np.ndarray, fit_dim: int = 3) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < fit_dim:
        return np.empty((0, fit_dim), dtype=float)
    if pts.shape[1] == fit_dim:
        return pts
    centered = pts - np.nanmean(pts, axis=0)
    if not np.isfinite(centered).all():
        return np.empty((0, fit_dim), dtype=float)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:fit_dim].T


def _ellipse_distance(
    points: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    r1: float,
    r2: float,
) -> np.ndarray:
    dp = points - center
    along_n = (dp @ normal)[:, None] * normal
    perp = dp - along_n
    cu = perp @ u_axis
    cv = perp @ v_axis
    phi = np.arctan2(cv / (r2 + 1e-12), cu / (r1 + 1e-12))
    nearest = center + (r1 * np.cos(phi))[:, None] * u_axis + (r2 * np.sin(phi))[:, None] * v_axis
    return np.linalg.norm(points - nearest, axis=1)


def torus_geometry_features(segment: np.ndarray, fs: int = FS, tau: int = 20, embedding_dim: int = 3) -> np.ndarray:
    x = detrend_zscore(segment)
    try:
        x = bandpass_filter(x, fs=fs, low=2.0, high=55.0)
    except ValueError:
        pass
    points = torus_fit_coordinates(lag_embed(x, dim=embedding_dim, tau=tau))
    if len(points) < 12:
        return np.full(15, np.nan, dtype=float)

    center = points.mean(axis=0)
    centered = points - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    extents = 2.0 * singular_values / np.sqrt(len(points))
    minor = max(extents[2] / 2.0, 1e-9)
    r1 = max(extents[0] / 2.0 - minor, minor)
    r2 = max(extents[1] / 2.0 - minor, minor)
    u_axis, v_axis, normal = vt[0], vt[1], vt[2]
    dist = _ellipse_distance(points, center, normal, u_axis, v_axis, r1, r2)
    signed = dist - minor
    outside = np.maximum(0.0, signed)
    base = np.array(
        [
            r1,
            r2,
            minor,
            np.mean(outside**2),
            np.mean(outside),
            np.mean(signed <= 0.0),
        ],
        dtype=float,
    )
    return np.concatenate([base, normal, u_axis, v_axis])


def fit_elliptical_torus_3d(
    points: np.ndarray,
    lam: float = 1.0,
    lam_h: float = 0.5,
    n_modes: int = 3,
    max_nfev: int = 1200,
) -> dict[str, object]:
    pts = np.asarray(points, dtype=np.float64)
    n_points = len(pts)
    center0 = pts.mean(axis=0)
    centered = pts - center0
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    pc_extents = 2.0 * singular_values / np.sqrt(n_points)

    r_target = pc_extents[2] / 2.0
    r1_target = max(pc_extents[0] / 2.0 - r_target, r_target)
    r2_target = max(pc_extents[1] / 2.0 - r_target, r_target)
    u0, _v0, n0 = vt[0], vt[1], vt[2]

    def _build_frame(dn: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        r1 = np.linalg.norm(dn) + 1e-12
        n_ax = dn / r1
        u_cand = u0 - np.dot(u0, n_ax) * n_ax
        u_norm = np.linalg.norm(u_cand)
        if u_norm < 1e-10:
            u_cand = vt[1] - np.dot(vt[1], n_ax) * n_ax
            u_norm = np.linalg.norm(u_cand)
        u_ax = u_cand / (u_norm + 1e-12)
        v_ax = np.cross(n_ax, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return r1, n_ax, u_ax, v_ax

    def _ellipse_torus_distance(
        center: np.ndarray,
        n_ax: np.ndarray,
        u_ax: np.ndarray,
        v_ax: np.ndarray,
        r1: float,
        r2: float,
        fit_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        dp = fit_points - center
        along_n = (dp @ n_ax)[:, None] * n_ax
        perp = dp - along_n
        cu = perp @ u_ax
        cv = perp @ v_ax
        phi = np.arctan2(cv / (r2 + 1e-12), cu / (r1 + 1e-12))
        nearest = center + (r1 * np.cos(phi))[:, None] * u_ax + (r2 * np.sin(phi))[:, None] * v_ax
        return np.linalg.norm(fit_points - nearest, axis=1), phi

    def residuals(params: np.ndarray) -> np.ndarray:
        center = params[:3]
        dn = params[3:6]
        r1, n_ax, u_ax, v_ax = _build_frame(dn)
        r2 = r1 * np.exp(params[6])
        minor = np.exp(params[7])
        dist, phi = _ellipse_torus_distance(center, n_ax, u_ax, v_ax, r1, r2, pts)
        surface_resid = dist - minor
        w = lam * np.sqrt(n_points)
        reg = np.array(
            [
                w * (2.0 * (r1 + minor) - pc_extents[0]) / (pc_extents[0] + 1e-12),
                w * (2.0 * (r2 + minor) - pc_extents[1]) / (pc_extents[1] + 1e-12),
                w * (2.0 * minor - pc_extents[2]) / (pc_extents[2] + 1e-12),
            ]
        )
        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(n_points) * np.mean(surface_resid * np.cos(k * phi)))
            homog.append(lam_h * np.sqrt(n_points) * np.mean(surface_resid * np.sin(k * phi)))
        return np.concatenate([surface_resid, reg, np.asarray(homog)])

    ratio0 = np.clip(r2_target / (r1_target + 1e-12), 0.1, 1.0)
    x0 = np.concatenate([center0, n0 * r1_target, [np.log(ratio0)], [np.log(max(r_target, 1e-3))]])
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    max_r1 = max(pc_extents[0] / 2.0, 1e-3)
    lb[3:6] = -max_r1
    ub[3:6] = max_r1
    ratio_cap = np.clip(pc_extents[1] / (pc_extents[0] + 1e-12), 0.05, 1.0)
    lb[6] = np.log(0.05)
    ub[6] = np.log(ratio_cap)
    lb[7] = np.log(1e-3)
    ub[7] = np.log(pc_extents[2] / 2.0 + 1e-6)

    x0 = np.clip(x0, lb, ub)
    result = optimize.least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        loss="huber",
        f_scale=1.0,
        max_nfev=max_nfev,
    )

    center = result.x[:3]
    dn = result.x[3:6]
    r1, n_ax, u_ax, v_ax = _build_frame(dn)
    r2 = r1 * np.exp(result.x[6])
    minor = np.exp(result.x[7])
    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        r1, r2 = r2, r1
        u_ax, v_ax = v_ax, u_ax

    dist, _ = _ellipse_torus_distance(center, n_ax, u_ax, v_ax, r1, r2, pts)
    signed = dist - minor
    outside = np.maximum(0.0, signed)
    return {
        "center": center,
        "direction": n_ax,
        "u_axis": u_ax,
        "v_axis": v_ax,
        "R1": float(r1),
        "R2": float(r2),
        "minor_radius": float(minor),
        "mse": float(np.mean(outside**2)),
        "mean_error": float(np.mean(outside)),
        "frac_inside": float(np.mean(signed <= 0.0)),
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }


def nonlinear_torus_geometry_features(
    segment: np.ndarray,
    fs: int = FS,
    tau: int = 20,
    embedding_dim: int = 3,
    n_points: int = 300,
    max_nfev: int = 1200,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    x = detrend_zscore(segment)
    x = bandpass_filter(x, fs=fs, low=2.0, high=55.0)
    points = torus_fit_coordinates(lag_embed(x, dim=embedding_dim, tau=tau))
    if len(points) < 12:
        return np.full(15, np.nan, dtype=float), {"success": False, "reason": "too_few_points"}
    if len(points) > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=n_points, replace=False)
        points = points[np.sort(idx)]
    fit = fit_elliptical_torus_3d(points, max_nfev=max_nfev)
    features = np.concatenate(
        [
            np.array(
                [
                    fit["R1"],
                    fit["R2"],
                    fit["minor_radius"],
                    fit["mse"],
                    fit["mean_error"],
                    fit["frac_inside"],
                ],
                dtype=float,
            ),
            np.asarray(fit["direction"], dtype=float),
            np.asarray(fit["u_axis"], dtype=float),
            np.asarray(fit["v_axis"], dtype=float),
        ]
    )
    metadata = {"success": bool(fit["success"]), "cost": fit["cost"], "nfev": fit["nfev"]}
    return features, metadata


def extract_epoch_segments(data: np.lib.npyio.NpzFile, epoch_name: str) -> tuple[np.ndarray, np.ndarray]:
    lfp = data["lfp"].astype(np.float32)
    movement_onset_ms = data["movement_onset_ms"].astype(float)
    fs = int(data["sampling_rate_hz"])
    go_sample = int(data["go_sample"])
    anchor, start_s, end_s = EPOCHS[epoch_name]

    segments: list[np.ndarray] = []
    keep: list[int] = []
    for i, trial in enumerate(lfp):
        if anchor == "go":
            center = go_sample
        else:
            if not np.isfinite(movement_onset_ms[i]):
                continue
            center = int(round(go_sample + movement_onset_ms[i] * fs / 1000.0))
        start = int(round(center + start_s * fs))
        end = int(round(center + end_s * fs))
        if start < 0 or end > len(trial) or end <= start:
            continue
        segment = trial[start:end]
        if np.isfinite(segment).all():
            segments.append(segment.astype(np.float32, copy=False))
            keep.append(i)
    if not segments:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int)
    return np.vstack(segments), np.asarray(keep, dtype=int)


def condition_labels(direction: np.ndarray, delay_label: np.ndarray) -> np.ndarray:
    labels = []
    for d, delay in zip(direction, delay_label):
        if int(d) in DIRECTION_LABELS and str(delay) in DELAY_LABELS:
            labels.append(f"D{int(d)}_{delay}")
        else:
            labels.append("unknown")
    return np.asarray(labels, dtype=object)


def target_values(data: np.lib.npyio.NpzFile, target: str, keep: np.ndarray) -> tuple[np.ndarray, list[Any]]:
    if target == "direction":
        labels = data["direction"].astype(int)[keep]
        valid = np.isin(labels, DIRECTION_LABELS)
        return labels[valid], DIRECTION_LABELS
    if target == "delay":
        labels = data["delay_label"].astype(str)[keep]
        valid = np.isin(labels, DELAY_LABELS)
        return labels[valid], DELAY_LABELS
    if target == "condition":
        labels_all = condition_labels(data["direction"].astype(int), data["delay_label"].astype(str))[keep]
        valid = labels_all != "unknown"
        classes = [f"D{d}_{delay}" for delay in DELAY_LABELS for d in DIRECTION_LABELS]
        return labels_all[valid], classes
    raise ValueError(f"Unknown target: {target}")


def valid_target_mask(data: np.lib.npyio.NpzFile, target: str, keep: np.ndarray) -> np.ndarray:
    if target == "direction":
        labels = data["direction"].astype(int)[keep]
        return np.isin(labels, DIRECTION_LABELS)
    if target == "delay":
        labels = data["delay_label"].astype(str)[keep]
        return np.isin(labels, DELAY_LABELS)
    if target == "condition":
        labels = condition_labels(data["direction"].astype(int), data["delay_label"].astype(str))[keep]
        return labels != "unknown"
    raise ValueError(f"Unknown target: {target}")


def balanced_indices(labels: np.ndarray, classes: Iterable[Any], seed: int = RANDOM_SEED) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    per_class = [np.where(labels == cls)[0] for cls in classes if np.sum(labels == cls) > 0]
    if len(per_class) < 2:
        return None
    min_count = min(len(idx) for idx in per_class)
    if min_count < MIN_CLASS_COUNT:
        return None
    chosen = [rng.choice(idx, size=min_count, replace=False) for idx in per_class]
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def decode_features(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[Any],
    standardize: bool = True,
) -> dict[str, Any]:
    present_classes = [cls for cls in classes if np.sum(labels == cls) > 0]
    idx = balanced_indices(labels, present_classes)
    if idx is None:
        return {
            "status": "insufficient_classes",
            "accuracy": np.nan,
            "f1": np.nan,
            "n_classes": len(present_classes),
            "n_trials_balanced": 0,
            "per_class_n": 0,
            "class_labels": present_classes,
            "confusion": np.full((len(present_classes), len(present_classes)), np.nan),
        }

    x_bal = np.asarray(features[idx], dtype=float)
    y_bal = np.asarray(labels[idx])
    finite = np.isfinite(x_bal).all(axis=1)
    x_bal = x_bal[finite]
    y_bal = y_bal[finite]
    class_labels = [cls for cls in present_classes if np.sum(y_bal == cls) > 0]
    counts = [np.sum(y_bal == cls) for cls in class_labels]
    if len(class_labels) < 2 or min(counts) < MIN_CLASS_COUNT:
        return {
            "status": "insufficient_classes",
            "accuracy": np.nan,
            "f1": np.nan,
            "n_classes": len(class_labels),
            "n_trials_balanced": int(len(y_bal)),
            "per_class_n": int(min(counts) if counts else 0),
            "class_labels": class_labels,
            "confusion": np.full((len(class_labels), len(class_labels)), np.nan),
        }

    col_std = np.nanstd(x_bal, axis=0)
    keep_cols = col_std > 1e-12
    if not np.any(keep_cols):
        return {
            "status": "degenerate_features",
            "accuracy": np.nan,
            "f1": np.nan,
            "n_classes": len(class_labels),
            "n_trials_balanced": int(len(y_bal)),
            "per_class_n": int(min(np.sum(y_bal == cls) for cls in class_labels)),
            "class_labels": class_labels,
            "confusion": np.full((len(class_labels), len(class_labels)), np.nan),
        }
    x_bal = x_bal[:, keep_cols]

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) if standardize else LinearDiscriminantAnalysis()
    try:
        pred = cross_val_predict(estimator, x_bal, y_bal, cv=cv)
    except (IndexError, ValueError, np.linalg.LinAlgError):
        return {
            "status": "degenerate_features",
            "accuracy": np.nan,
            "f1": np.nan,
            "n_classes": len(class_labels),
            "n_trials_balanced": int(len(y_bal)),
            "per_class_n": int(min(np.sum(y_bal == cls) for cls in class_labels)),
            "class_labels": class_labels,
            "confusion": np.full((len(class_labels), len(class_labels)), np.nan),
        }
    cm = confusion_matrix(y_bal, pred, labels=class_labels, normalize="true")
    return {
        "status": "ok",
        "accuracy": float(accuracy_score(y_bal, pred)),
        "f1": float(f1_score(y_bal, pred, average="macro", labels=class_labels)),
        "n_classes": len(class_labels),
        "n_trials_balanced": int(len(y_bal)),
        "per_class_n": int(min(np.sum(y_bal == cls) for cls in class_labels)),
        "class_labels": class_labels,
        "confusion": cm,
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
