"""Fixed m=3, tau=16 point clouds and existing geometric fits; no decoder."""
from __future__ import annotations

import hashlib
import importlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

from motor_lfp_utils import lag_embed
from select_trace_embedding_parameters import preprocess_segments

UNIT = Path(__file__).resolve().parent
RECORDING = "monkeyT_session-y070316009-12_lfp-7"
INPUT = UNIT / "outputs/recording_198_trial_dossiers_v1" / RECORDING / "inputs"
OUTPUT = UNIT / "outputs/prego_T_y070316009_12_ch7_tau16_geometric_fits_v1"
MODELS = ("one_torus", "two_torus")
NATIVE_METRICS = ("mse", "mean_error", "frac_inside", "r_squared", "hole_quality", "frac_in_hole")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode()+array.tobytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(serializable(value), indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def make_clouds(segments):
    segments = np.asarray(segments)
    if segments.ndim != 2 or segments.shape[1] != 500 or not np.isfinite(segments).all():
        raise ValueError("Expected finite trial-by-500-sample pre-GO epochs")
    if np.any(np.ptp(segments, axis=1) <= 1e-12):
        raise ValueError("Constant trial")
    processed = preprocess_segments(segments)
    clouds = np.stack([lag_embed(x, dim=3, tau=16) for x in processed])
    if clouds.shape != (len(segments), 468, 3):
        raise ValueError("Unexpected fixed-delay shape")
    return processed, clouds


def _ellipse_distance(cu, cv, a, b):
    # Reproduce the native four-iteration nearest-angle approximation exactly.
    phi = np.arctan2(cv*a, cu*b)
    for _ in range(4):
        cp, sp = np.cos(phi), np.sin(phi)
        dx, dy = cu-a*cp, cv-b*sp
        gradient = dx*a*sp-dy*b*cp
        hessian = dx*a*cp+dy*b*sp+(a*sp)**2+(b*cp)**2
        phi -= np.clip(gradient/(hessian+1e-12), -.5, .5)
    return np.sqrt((cu-a*np.cos(phi))**2+(cv-b*np.sin(phi))**2)


def recompute_distances(points, fit, model):
    dp = np.asarray(points)-np.asarray(fit["center"])
    h = dp @ np.asarray(fit["direction"])
    if model == "two_torus":
        perp = dp-h[:, None]*np.asarray(fit["direction"])
        cu, cv = perp @ fit["u_axis"], perp @ fit["v_axis"]
        radial = _ellipse_distance(cu, cv, fit["R1"], fit["R2"])
        return np.maximum(0., np.sqrt(radial**2+h**2)-fit["minor_radius"])
    if model != "one_torus":
        raise ValueError(model)
    cu, cv = dp @ fit["u_axis"], dp @ fit["v_axis"]
    outer = (cu/(fit["R1"]+1e-12))**2+(cv/(fit["R2"]+1e-12))**2 > 1
    inner = (cu/(fit["R1_in"]+1e-12))**2+(cv/(fit["R2_in"]+1e-12))**2 < 1
    radial = np.zeros(len(dp))
    radial[outer] = _ellipse_distance(cu, cv, fit["R1"], fit["R2"])[outer]
    radial[inner] = _ellipse_distance(cu, cv, fit["R1_in"], fit["R2_in"])[inner]
    return np.sqrt(radial**2+h**2)


def summarize_fit(points, fit, model):
    diag = fit["diagnostics"]
    distance = np.asarray(diag["distance_3d"], dtype=float)
    denominator = float(np.sum((points-points.mean(axis=0))**2))
    values = np.r_[distance, fit["R1"], fit["R2"], fit["minor_radius"],
                   *[fit[k] for k in NATIVE_METRICS], fit["center"], fit["direction"],
                   fit["u_axis"], fit["v_axis"]]
    flags = []
    if not diag["success"]:
        flags.append("nonconvergence")
    if not np.isfinite(values).all() or denominator <= 1e-20:
        flags.append("nonfinite_or_zero_variance")
    if fit["R1"] <= 1e-8 or fit["R2"] <= 1e-8:
        flags.append("degenerate_radius")
    if model == "one_torus":
        if min(fit["R1_in"], fit["R2_in"]) <= 1e-8:
            flags.append("collapsed_hole")
        if fit["R1_in"] >= fit["R1"] or fit["R2_in"] >= fit["R2"]:
            flags.append("invalid_inner_outer_radii")
    else:
        if fit["minor_radius"] >= min(fit["R1"], fit["R2"]):
            flags.append("collapsed_hole")
        if fit["minor_radius"] <= 1e-8:
            flags.append("collapsed_tube")
    if np.any(np.asarray(diag["active_mask"]) != 0):
        flags.append("active_optimizer_bound")
    params = np.asarray(diag["params"], dtype=float)
    lower = np.asarray(diag["lower_bounds"], dtype=float)
    upper = np.asarray(diag["upper_bounds"], dtype=float)
    near = ((np.isfinite(lower) & (np.abs(params-lower) <= 1e-5*(1+np.abs(params)))) |
            (np.isfinite(upper) & (np.abs(params-upper) <= 1e-5*(1+np.abs(params)))))
    if np.any(near):
        flags.append("near_optimizer_bound")
    ratio = max(fit["R1"], fit["R2"])/max(min(fit["R1"], fit["R2"]), 1e-30)
    if ratio < 1.05:
        flags.append("nearly_circular_orientation")
    frame = np.column_stack([fit["u_axis"], fit["v_axis"], fit["direction"]])
    if np.max(np.abs(frame.T @ frame-np.eye(3))) > 1e-5:
        flags.append("degenerate_orientation")
    row = dict(model=model, n_points=len(points), optimizer_success=bool(diag["success"]),
               optimizer_status=diag["status"], optimizer_message=diag["message"],
               nfev=diag["nfev"], cost=diag["cost"], optimality=diag["optimality"],
               flags=";".join(flags), cloud_variance_sum=denominator,
               normalized_3d_set_error=float(np.sum(distance**2)/denominator),
               rms_3d_set_distance=float(np.sqrt(np.mean(distance**2))),
               mean_3d_set_distance=float(np.mean(distance)),
               active_bounds=int(np.count_nonzero(diag["active_mask"])), near_bounds=int(near.sum()))
    row.update({f"native_{key}": fit[key] for key in NATIVE_METRICS})
    for key in ("R1", "R2", "minor_radius", "R1_in", "R2_in"):
        row[key] = fit.get(key, np.nan)
    for key in ("center", "direction", "u_axis", "v_axis"):
        row.update({f"{key}_{axis}": value for axis, value in zip("xyz", fit[key])})
    return row


def fit_one(points, model):
    started = time.monotonic()
    if model not in MODELS:
        raise ValueError(model)
    result = dict(model=model, cloud_sha256=array_hash(points))
    try:
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite Nx3 cloud")
        if len(points) < 12 or np.sum(np.var(points, axis=0)) <= 1e-20:
            raise ValueError("Degenerate cloud")
        module = importlib.import_module(f"NeuralFieldManifold.fits.{model}")
        fitter = getattr(module, f"{model}_fit")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fitter(points, lam=.1, hole_ratio=.5 if model == "one_torus" else .75,
                         n_modes=3, return_diagnostics=True)
        summary = summarize_fit(points, fit, model)
        result.update(status="returned", reason="", fit=fit, summary=summary,
                      warnings=[str(w.message) for w in caught])
    except Exception as error:
        result.update(status="failed", reason=f"{type(error).__name__}: {error}", warnings=[])
    result["elapsed_seconds"] = time.monotonic()-started
    return result


def example_rows(trials):
    return trials.sort_values(["direction", "original_trial_number", "row_index"]).groupby(
        "direction", sort=True).first()["row_index"].astype(int).tolist()
