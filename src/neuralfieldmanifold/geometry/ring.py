import numpy as np
from scipy import optimize

from .results import FitConfig, GeometryFit, validate_cloud


def fit_ring(points, *, config=None):
    """Fit an elliptical ring using native containment and compactness residuals.

    Accepts a finite (N, 3) cloud. No scaling or projection is applied.
    Returns a GeometryFit with optimizer status and native geometric measures."""
    config = FitConfig() if config is None else config
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig")
    lam = config.regularization
    hole_ratio = 0.5 if config.hole_ratio is None else config.hole_ratio
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
    (u0, v0, n0) = (Vt[0], Vt[1], Vt[2])
    R1_out_tgt = pc_ext[0] / 2.0
    R2_out_tgt = pc_ext[1] / 2.0
    proj_cu = centered @ u0
    proj_cv = centered @ v0
    rho = np.sqrt(
        (proj_cu / (R1_out_tgt + 1e-12)) ** 2 + (proj_cv / (R2_out_tgt + 1e-12)) ** 2
    )
    rho_in = np.percentile(rho, 15)
    R1_in_tgt = max(R1_out_tgt * rho_in, 0.001)

    def _frame(dn):
        R1_out = np.linalg.norm(dn) + 1e-12
        n_ax = dn / R1_out
        u = u0 - np.dot(u0, n_ax) * n_ax
        u_ax = u / (np.linalg.norm(u) + 1e-12)
        v_ax = np.cross(n_ax, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return (R1_out, n_ax, u_ax, v_ax)

    def _nearest_phi(cu, cv, R1, R2, iters=4):
        phi = np.arctan2(cv * R1, cu * R2)
        for _ in range(iters):
            (cp, sp) = (np.cos(phi), np.sin(phi))
            (dx, dy) = (cu - R1 * cp, cv - R2 * sp)
            g = dx * R1 * sp - dy * R2 * cp
            H = dx * R1 * cp + dy * R2 * sp + (R1 * sp) ** 2 + (R2 * cp) ** 2
            phi -= np.clip(g / (H + 1e-12), -0.5, 0.5)
        return phi

    def _ellipse_dist_2d(cu, cv, R1, R2):
        phi = _nearest_phi(cu, cv, R1, R2)
        near_u = R1 * np.cos(phi)
        near_v = R2 * np.sin(phi)
        return (np.sqrt((cu - near_u) ** 2 + (cv - near_v) ** 2), phi)

    def residuals(params):
        c = params[:3]
        dn = params[3:6]
        (R1_out, n_ax, u_ax, v_ax) = _frame(dn)
        ratio = np.exp(params[6])
        R2_out = R1_out * ratio
        R1_in = np.exp(params[7])
        R2_in = R1_in * ratio
        dp = pts - c
        h = dp @ n_ax
        cu = dp @ u_ax
        cv = dp @ v_ax
        sigma_out_sq = (cu / (R1_out + 1e-12)) ** 2 + (cv / (R2_out + 1e-12)) ** 2
        sigma_in_sq = (cu / (R1_in + 1e-12)) ** 2 + (cv / (R2_in + 1e-12)) ** 2
        outside_outer = sigma_out_sq > 1.0
        inside_inner = sigma_in_sq < 1.0
        (d_out, _) = _ellipse_dist_2d(cu, cv, R1_out, R2_out)
        (d_in, _) = _ellipse_dist_2d(cu, cv, R1_in, R2_in)
        coverage_2d = np.zeros(N)
        coverage_2d[outside_outer] = d_out[outside_outer]
        coverage_2d[inside_inner] = d_in[inside_inner]
        coverage = np.sqrt(coverage_2d**2 + h**2)
        w = lam * np.sqrt(N)
        compact = np.array(
            [
                w * max(0.0, 2 * R1_out - pc_ext[0]) / (pc_ext[0] + 1e-12),
                w * max(0.0, 2 * R2_out - pc_ext[1]) / (pc_ext[1] + 1e-12),
                3.0 * w * max(0.0, hole_ratio - R1_in / (R1_out + 1e-12)),
                3.0 * w * max(0.0, R1_in - R1_out) / (R1_out + 1e-12),
            ]
        )
        phi = np.arctan2(cv * R1_out, cu * R2_out)
        lam_h = 0.3
        raw = np.zeros(N)
        raw[outside_outer] = d_out[outside_outer]
        raw[inside_inner] = -d_in[inside_inner]
        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.cos(k * phi)))
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.sin(k * phi)))
        return np.concatenate([coverage, compact, np.array(homog)])

    ratio0 = np.clip(R2_out_tgt / (R1_out_tgt + 1e-12), 0.1, 1.0)
    x0 = np.concatenate(
        [center0, n0 * R1_out_tgt, [np.log(ratio0)], [np.log(max(R1_in_tgt, 0.001))]]
    )
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    max_R1 = pc_range[0] / 2.0
    (lb[3:6], ub[3:6]) = (-max_R1, max_R1)
    lb[6] = np.log(0.05)
    ub[6] = np.log(np.clip(pc_range[1] / (pc_range[0] + 1e-12), 0.05, 1.0))
    lb[7] = np.log(0.001)
    ub[7] = np.log(max_R1 + 1e-06)
    x0 = np.clip(x0, lb + 1e-08, ub - 1e-08)
    result = optimize.least_squares(
        residuals, x0, bounds=(lb, ub), loss="huber", f_scale=1.0, max_nfev=6000
    )
    c = result.x[:3]
    dn = result.x[3:6]
    (R1_out, n_ax, u_ax, v_ax) = _frame(dn)
    ratio = np.exp(result.x[6])
    R2_out = R1_out * ratio
    R1_in = np.exp(result.x[7])
    R2_in = R1_in * ratio
    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        (R1_out, R2_out) = (R2_out, R1_out)
        (R1_in, R2_in) = (R2_in, R1_in)
        (u_ax, v_ax) = (v_ax, u_ax)
    dp = pts - c
    h = dp @ n_ax
    cu = dp @ u_ax
    cv = dp @ v_ax
    sigma_out_sq = (cu / (R1_out + 1e-12)) ** 2 + (cv / (R2_out + 1e-12)) ** 2
    sigma_in_sq = (cu / (R1_in + 1e-12)) ** 2 + (cv / (R2_in + 1e-12)) ** 2
    outside_outer = sigma_out_sq > 1.0
    inside_inner = sigma_in_sq < 1.0
    in_band = ~outside_outer & ~inside_inner
    (d_out, _) = _ellipse_dist_2d(cu, cv, R1_out, R2_out)
    (d_in, _) = _ellipse_dist_2d(cu, cv, R1_in, R2_in)
    error_2d = np.zeros(N)
    error_2d[outside_outer] = d_out[outside_outer]
    error_2d[inside_inner] = d_in[inside_inner]
    error_3d = np.abs(h)
    error_3d[outside_outer] = np.sqrt(d_out[outside_outer] ** 2 + h[outside_outer] ** 2)
    error_3d[inside_inner] = np.sqrt(d_in[inside_inner] ** 2 + h[inside_inner] ** 2)
    ss_res = np.sum(error_2d**2)
    ss_tot = np.sum(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)
    output = dict(
        center=c,
        direction=n_ax,
        u_axis=u_ax,
        v_axis=v_ax,
        R1=float(R1_out),
        R2=float(R2_out),
        R1_in=float(R1_in),
        R2_in=float(R2_in),
        minor_radius=float((R1_out - R1_in + R2_out - R2_in) / 4),
        hole_quality=float(R1_in / (R1_out + 1e-12)),
        r_squared=float(r_squared),
        mse=float(np.mean(error_3d**2)),
        mean_error=float(np.mean(error_2d)),
        frac_inside=float(in_band.sum() / N),
        frac_in_hole=float(inside_inner.sum() / N),
        pc_extents=pc_ext,
        pc_ranges=pc_range,
    )
    return GeometryFit(
        kind="ring",
        center=c,
        normal=n_ax,
        u_axis=u_ax,
        v_axis=v_ax,
        major_radii=(output["R1"], output["R2"]),
        tube_radius=output["minor_radius"],
        inner_radii=(output["R1_in"], output["R2_in"]),
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
