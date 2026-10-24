import numpy as np
from scipy import optimize

from .results import FitConfig, GeometryFit, validate_cloud


def fit_torus(points, *, config=None):
    """Fit an elliptical torus using native containment and compactness residuals.

    Accepts a finite (N, 3) cloud. No scaling or projection is applied.
    Returns a GeometryFit with optimizer status and native geometric measures."""
    config = FitConfig() if config is None else config
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig")
    lam = config.regularization
    hole_ratio = 0.75 if config.hole_ratio is None else config.hole_ratio
    n_modes = config.angular_modes
    points = validate_cloud(points)
    pts = np.asarray(points, dtype=np.float64)
    N = len(pts)
    center0 = pts.mean(axis=0)
    centered = pts - center0
    (_, S, Vt) = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt.T
    pc_ext = np.array(
        [
            np.percentile(proj[:, i], 97.5) - np.percentile(proj[:, i], 2.5)
            for i in range(3)
        ]
    )
    pc_range = np.ptp(proj, axis=0)
    r_tgt = pc_ext[2] / 2.0
    R1_tgt = max(pc_ext[0] / 2.0 - r_tgt, r_tgt * 0.5)
    R2_tgt = max(pc_ext[1] / 2.0 - r_tgt, r_tgt * 0.5)
    (u0, v0, n0) = (Vt[0], Vt[1], Vt[2])

    def _frame(dn):
        R1 = np.linalg.norm(dn) + 1e-12
        n_ax = dn / R1
        u = u0 - np.dot(u0, n_ax) * n_ax
        u_ax = u / (np.linalg.norm(u) + 1e-12)
        v_ax = np.cross(n_ax, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return (R1, n_ax, u_ax, v_ax)

    def _nearest_phi(cu, cv, R1, R2, iters=4):
        phi = np.arctan2(cv * R1, cu * R2)
        for _ in range(iters):
            (cp, sp) = (np.cos(phi), np.sin(phi))
            (dx, dy) = (cu - R1 * cp, cv - R2 * sp)
            g = dx * R1 * sp - dy * R2 * cp
            h = dx * R1 * cp + dy * R2 * sp + (R1 * sp) ** 2 + (R2 * cp) ** 2
            phi -= np.clip(g / (h + 1e-12), -0.5, 0.5)
        return phi

    def _dist(c, n_ax, u_ax, v_ax, R1, R2, pts):
        dp = pts - c
        h = dp @ n_ax
        perp = dp - h[:, None] * n_ax
        (cu, cv) = (perp @ u_ax, perp @ v_ax)
        phi = _nearest_phi(cu, cv, R1, R2)
        near = (
            (R1 * np.cos(phi))[:, None] * u_ax + (R2 * np.sin(phi))[:, None] * v_ax + c
        )
        return (np.linalg.norm(pts - near, axis=1), phi)

    def residuals(params):
        c = params[:3]
        dn = params[3:6]
        (R1, n_ax, u_ax, v_ax) = _frame(dn)
        R2 = R1 * np.exp(params[6])
        r = np.exp(params[7])
        (d, phi) = _dist(c, n_ax, u_ax, v_ax, R1, R2, pts)
        coverage = np.maximum(0.0, d - r)
        w = lam * np.sqrt(N)
        Rmin = min(R1, R2)
        hole_overshoot = max(0.0, r - hole_ratio * Rmin) / (Rmin + 1e-12)
        compact = np.array(
            [
                w * max(0.0, 2 * (R1 + r) - pc_ext[0]) / (pc_ext[0] + 1e-12),
                w * max(0.0, 2 * (R2 + r) - pc_ext[1]) / (pc_ext[1] + 1e-12),
                w * max(0.0, 2 * r - pc_ext[2]) / (pc_ext[2] + 1e-12),
                3.0 * w * hole_overshoot,
            ]
        )
        lam_h = 0.3
        raw = d - r
        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.cos(k * phi)))
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.sin(k * phi)))
        return np.concatenate([coverage, compact, np.array(homog)])

    ratio0 = np.clip(R2_tgt / (R1_tgt + 1e-12), 0.1, 1.0)
    x0 = np.concatenate(
        [center0, n0 * R1_tgt, [np.log(ratio0)], [np.log(max(r_tgt, 0.001))]]
    )
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    max_R1 = pc_range[0] / 2.0
    (lb[3:6], ub[3:6]) = (-max_R1, max_R1)
    lb[6] = np.log(0.05)
    ub[6] = np.log(np.clip(pc_range[1] / (pc_range[0] + 1e-12), 0.05, 1.0))
    lb[7] = np.log(0.001)
    ub[7] = np.log(pc_range[2] / 2.0 + 1e-06)
    x0 = np.clip(x0, lb + 1e-08, ub - 1e-08)
    result = optimize.least_squares(
        residuals, x0, bounds=(lb, ub), loss="huber", f_scale=1.0, max_nfev=6000
    )
    c = result.x[:3]
    dn = result.x[3:6]
    (R1, n_ax, u_ax, v_ax) = _frame(dn)
    R2 = R1 * np.exp(result.x[6])
    r = np.exp(result.x[7])
    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        (R1, R2) = (R2, R1)
        (u_ax, v_ax) = (v_ax, u_ax)
    (d, _) = _dist(c, n_ax, u_ax, v_ax, R1, R2, pts)
    signed = d - r
    outside_dist = np.maximum(0.0, signed)
    inside_mask = signed <= 0
    dp = pts - c
    perp = dp - (dp @ n_ax)[:, None] * n_ax
    ell = (perp @ u_ax / (R1 + 1e-12)) ** 2 + (perp @ v_ax / (R2 + 1e-12)) ** 2
    in_hole = (ell < 1.0) & ~inside_mask
    ss_res = np.sum(outside_dist**2)
    ss_tot = np.sum(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    output = dict(
        center=c,
        direction=n_ax,
        u_axis=u_ax,
        v_axis=v_ax,
        R1=float(R1),
        R2=float(R2),
        minor_radius=float(r),
        hole_quality=float(1.0 - r / (min(R1, R2) + 1e-12)),
        r_squared=float(r_squared),
        mse=float(np.mean(outside_dist**2)),
        mean_error=float(np.mean(outside_dist)),
        frac_inside=float(inside_mask.sum() / N),
        frac_in_hole=float(in_hole.sum() / N),
        pc_extents=pc_ext,
        pc_ranges=pc_range,
    )
    return GeometryFit(
        kind="torus",
        center=c,
        normal=n_ax,
        u_axis=u_ax,
        v_axis=v_ax,
        major_radii=(output["R1"], output["R2"]),
        tube_radius=output["minor_radius"],
        inner_radii=None,
        score=output["frac_inside"],
        error=output["mean_error"],
        mse=output["mse"],
        r_squared=output["r_squared"],
        hole_quality=output["hole_quality"],
        fraction_in_hole=output["frac_in_hole"],
        success=bool(result.success),
        evaluations=int(result.nfev),
        optimizer_message=str(result.message),
    )
