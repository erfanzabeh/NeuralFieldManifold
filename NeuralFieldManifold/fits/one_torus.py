import numpy as np
from scipy import optimize
from plotly import graph_objects as go


def one_torus_fit(points, lam=1.0, hole_ratio=0.5, n_modes=3):
    """
    Fit an elliptical annular band (one-torus) to a 3-D point cloud.

    Objective
    ─────────
    COVERAGE    : Σ dᵢ²  for points outside the band (2-D projected)
    COMPACTNESS : outer ellipse ≤ PCA extents, inner ≥ hole_ratio × outer

    The band is the region between an inner and outer ellipse lying in the
    best-fit plane.  Both ellipses share the same R1/R2 aspect ratio.

    Parameters
    ──────────
    points     : (N, 3) array
    lam        : tradeoff (0 = pure containment, ∞ = locked to PCA)
    hole_ratio : inner radius penalised when R1_in < hole_ratio * R1
                 0.5 keeps a clear hole without being overly restrictive
    n_modes    : Fourier modes for angular-homogeneity penalty
    """
    pts = np.asarray(points, dtype=np.float64)
    N   = len(pts)

    # ── PCA ──────────────────────────────────────────────────────────
    center0  = pts.mean(axis=0)
    centered = pts - center0
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)

    proj     = centered @ Vt.T
    pc_ext   = np.array([
        np.percentile(proj[:, i], 97.5) - np.percentile(proj[:, i], 2.5)
        for i in range(3)
    ])
    pc_range = np.ptp(proj, axis=0)

    u0, v0, n0 = Vt[0], Vt[1], Vt[2]

    # ── initial estimates ────────────────────────────────────────────
    R1_out_tgt = pc_ext[0] / 2.0
    R2_out_tgt = pc_ext[1] / 2.0

    # estimate inner from projected radial distribution
    proj_cu = centered @ u0
    proj_cv = centered @ v0
    rho = np.sqrt((proj_cu / (R1_out_tgt + 1e-12))**2
                + (proj_cv / (R2_out_tgt + 1e-12))**2)
    rho_in  = np.percentile(rho, 15)
    R1_in_tgt = max(R1_out_tgt * rho_in, 1e-3)

    # ── helpers ──────────────────────────────────────────────────────
    def _frame(dn):
        R1_out = np.linalg.norm(dn) + 1e-12
        n_ax   = dn / R1_out
        u      = u0 - np.dot(u0, n_ax) * n_ax
        u_ax   = u / (np.linalg.norm(u) + 1e-12)
        v_ax   = np.cross(n_ax, u_ax)
        v_ax   = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return R1_out, n_ax, u_ax, v_ax

    def _nearest_phi(cu, cv, R1, R2, iters=4):
        phi = np.arctan2(cv * R1, cu * R2)
        for _ in range(iters):
            cp, sp = np.cos(phi), np.sin(phi)
            dx, dy = cu - R1 * cp, cv - R2 * sp
            g = dx * R1 * sp - dy * R2 * cp
            H = dx * R1 * cp + dy * R2 * sp + (R1*sp)**2 + (R2*cp)**2
            phi -= np.clip(g / (H + 1e-12), -0.5, 0.5)
        return phi

    def _ellipse_dist_2d(cu, cv, R1, R2):
        phi    = _nearest_phi(cu, cv, R1, R2)
        near_u = R1 * np.cos(phi)
        near_v = R2 * np.sin(phi)
        return np.sqrt((cu - near_u)**2 + (cv - near_v)**2), phi

    # ── residuals ────────────────────────────────────────────────────
    def residuals(params):
        c  = params[:3]
        dn = params[3:6]
        R1_out, n_ax, u_ax, v_ax = _frame(dn)
        ratio  = np.exp(params[6])
        R2_out = R1_out * ratio
        R1_in  = np.exp(params[7])
        R2_in  = R1_in * ratio

        dp = pts - c
        h  = dp @ n_ax            # normal (z) offset from the plane
        cu = dp @ u_ax
        cv = dp @ v_ax

        # classify by normalised elliptic radius
        sigma_out_sq = (cu / (R1_out + 1e-12))**2 + (cv / (R2_out + 1e-12))**2
        sigma_in_sq  = (cu / (R1_in  + 1e-12))**2 + (cv / (R2_in  + 1e-12))**2
        outside_outer = sigma_out_sq > 1.0
        inside_inner  = sigma_in_sq  < 1.0

        d_out, _ = _ellipse_dist_2d(cu, cv, R1_out, R2_out)
        d_in,  _ = _ellipse_dist_2d(cu, cv, R1_in,  R2_in)

        # COVERAGE: 2D in-plane + normal offset so optimizer keeps
        # the plane aligned with the data in 3D
        coverage_2d = np.zeros(N)
        coverage_2d[outside_outer] = d_out[outside_outer]
        coverage_2d[inside_inner]  = d_in[inside_inner]
        coverage = np.sqrt(coverage_2d**2 + h**2)

        # COMPACTNESS
        w = lam * np.sqrt(N)
        compact = np.array([
            # outer ≤ PC extents
            w * max(0.0, 2*R1_out - pc_ext[0]) / (pc_ext[0] + 1e-12),
            w * max(0.0, 2*R2_out - pc_ext[1]) / (pc_ext[1] + 1e-12),
            # hole: inner ≥ hole_ratio × outer
            3.0 * w * max(0.0, hole_ratio - R1_in / (R1_out + 1e-12)),
            # safety: inner must stay < outer
            3.0 * w * max(0.0, R1_in - R1_out) / (R1_out + 1e-12),
        ])

        # HOMOGENEITY: keep band width uniform around the ring
        phi   = np.arctan2(cv * R1_out, cu * R2_out)
        lam_h = 0.3
        raw   = np.zeros(N)
        raw[outside_outer] = d_out[outside_outer]
        raw[inside_inner]  = -d_in[inside_inner]

        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.cos(k*phi)))
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.sin(k*phi)))

        return np.concatenate([coverage, compact, np.array(homog)])

    # ── init & bounds ────────────────────────────────────────────────
    ratio0 = np.clip(R2_out_tgt / (R1_out_tgt + 1e-12), 0.1, 1.0)
    x0 = np.concatenate([
        center0,
        n0 * R1_out_tgt,           # dn: ||dn|| = R1_out, direction = normal
        [np.log(ratio0)],
        [np.log(max(R1_in_tgt, 1e-3))],
    ])

    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0,  np.inf)
    max_R1 = pc_range[0] / 2.0
    lb[3:6], ub[3:6] = -max_R1, max_R1
    lb[6] = np.log(0.05)
    ub[6] = np.log(np.clip(pc_range[1]/(pc_range[0]+1e-12), 0.05, 1.0))
    lb[7] = np.log(1e-3)
    ub[7] = np.log(max_R1 + 1e-6)

    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    result = optimize.least_squares(
        residuals, x0,
        bounds=(lb, ub),
        loss='huber', f_scale=1.0,
        max_nfev=6000,
    )

    # ── extract ──────────────────────────────────────────────────────
    c  = result.x[:3]
    dn = result.x[3:6]
    R1_out, n_ax, u_ax, v_ax = _frame(dn)
    ratio  = np.exp(result.x[6])
    R2_out = R1_out * ratio
    R1_in  = np.exp(result.x[7])
    R2_in  = R1_in * ratio

    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        R1_out, R2_out = R2_out, R1_out
        R1_in,  R2_in  = R2_in,  R1_in
        u_ax, v_ax = v_ax, u_ax

    # classify final
    dp = pts - c
    h  = dp @ n_ax                     # normal (z) offset from the plane
    cu = dp @ u_ax
    cv = dp @ v_ax
    sigma_out_sq = (cu/(R1_out+1e-12))**2 + (cv/(R2_out+1e-12))**2
    sigma_in_sq  = (cu/(R1_in+1e-12))**2  + (cv/(R2_in+1e-12))**2
    outside_outer = sigma_out_sq > 1.0
    inside_inner  = sigma_in_sq  < 1.0
    in_band       = (~outside_outer) & (~inside_inner)

    d_out, _ = _ellipse_dist_2d(cu, cv, R1_out, R2_out)
    d_in,  _ = _ellipse_dist_2d(cu, cv, R1_in,  R2_in)

    # 2D in-plane error (mean_error)
    error_2d = np.zeros(N)
    error_2d[outside_outer] = d_out[outside_outer]
    error_2d[inside_inner]  = d_in[inside_inner]

    # 3D error including normal offset (mse)
    error_3d = np.abs(h)                               # in-band: just |h|
    error_3d[outside_outer] = np.sqrt(d_out[outside_outer]**2 + h[outside_outer]**2)
    error_3d[inside_inner]  = np.sqrt(d_in[inside_inner]**2  + h[inside_inner]**2)

    # R² using 2D residuals (mean_error convention)
    ss_res = np.sum(error_2d**2)
    ss_tot = np.sum(np.sum((pts - pts.mean(axis=0))**2, axis=1))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

    return dict(
        center=c, direction=n_ax, u_axis=u_ax, v_axis=v_ax,
        R1=float(R1_out), R2=float(R2_out),
        R1_in=float(R1_in), R2_in=float(R2_in),
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


def plot_one_torus_fit(entry, n_surf=60, alpha_surf=0.15):

    pts    = np.asarray(entry['points'])
    center = np.asarray(entry['center'])
    axis   = np.asarray(entry['direction'])
    u_ax   = np.asarray(entry['u_axis'])
    v_ax   = np.asarray(entry['v_axis'])
    R1_out = float(entry['R1'])
    R2_out = float(entry['R2'])
    R1_in  = float(entry['R1_in'])
    R2_in  = float(entry['R2_in'])

    # classify inside / outside band
    dp = pts - center
    cu = dp @ u_ax
    cv = dp @ v_ax
    sigma_out = (cu/(R1_out+1e-12))**2 + (cv/(R2_out+1e-12))**2
    sigma_in  = (cu/(R1_in+1e-12))**2  + (cv/(R2_in+1e-12))**2
    inside = (sigma_out <= 1.0) & (sigma_in >= 1.0)

    # annular band mesh (flat surface between inner and outer ellipses)
    phi      = np.linspace(0, 2 * np.pi, n_surf)
    n_radial = 10
    t        = np.linspace(0, 1, n_radial)
    PHI, T   = np.meshgrid(phi, t)

    R1 = R1_in + T * (R1_out - R1_in)
    R2 = R2_in + T * (R2_out - R2_in)
    bx = R1 * np.cos(PHI)
    by = R2 * np.sin(PHI)

    X = center[0] + bx*u_ax[0] + by*v_ax[0]
    Y = center[1] + bx*u_ax[1] + by*v_ax[1]
    Z = center[2] + bx*u_ax[2] + by*v_ax[2]

    fig = go.Figure()

    # band surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=alpha_surf,
        colorscale=[[0, 'steelblue'], [1, 'steelblue']],
        showscale=False, name='Band',
    ))

    # outer ellipse curve
    phi_c = np.linspace(0, 2 * np.pi, n_surf)
    ox = R1_out * np.cos(phi_c)
    oy = R2_out * np.sin(phi_c)
    fig.add_trace(go.Scatter3d(
        x=center[0] + ox*u_ax[0] + oy*v_ax[0],
        y=center[1] + ox*u_ax[1] + oy*v_ax[1],
        z=center[2] + ox*u_ax[2] + oy*v_ax[2],
        mode='lines', line=dict(color='steelblue', width=2, dash='dash'),
        name='outer', showlegend=True,
    ))

    # inner ellipse curve
    ix = R1_in * np.cos(phi_c)
    iy = R2_in * np.sin(phi_c)
    fig.add_trace(go.Scatter3d(
        x=center[0] + ix*u_ax[0] + iy*v_ax[0],
        y=center[1] + ix*u_ax[1] + iy*v_ax[1],
        z=center[2] + ix*u_ax[2] + iy*v_ax[2],
        mode='lines', line=dict(color='coral', width=2, dash='dash'),
        name='inner', showlegend=True,
    ))

    # trajectory segments colored by inside/outside
    first_in = first_out = True
    i = 0
    n = len(pts)
    n_in, n_out = int(inside.sum()), int((~inside).sum())
    while i < n:
        is_in = bool(inside[i])
        j = i + 1
        while j < n and bool(inside[j]) == is_in:
            j += 1
        seg = pts[i:min(j + 1, n)]
        color = 'black' if is_in else 'darkred'
        if is_in:
            show = first_in; first_in = False
            name = f'inside ({n_in})'
            grp  = 'inside'
        else:
            show = first_out; first_out = False
            name = f'outside ({n_out})'
            grp  = 'outside'
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode='lines', line=dict(color=color, width=3),
            name=name, showlegend=show, legendgroup=grp,
        ))
        i = j

    t_start, t_end = entry['start_sec'], entry['end_sec']
    fig.update_layout(
        title=dict(text=(
            f"Elliptical band {t_start:.1f}–{t_end:.1f} s | "
            f"R1={R1_out:.1f}  R2={R2_out:.1f}  "
            f"R1_in={R1_in:.1f}  R2_in={R2_in:.1f}  "
            f"MSE={entry['mse']:.2f}"
        ), font=dict(size=13)),
        scene=dict(
            xaxis_title='x(t)',
            yaxis_title='x(t-tau)',
            zaxis_title='x(t-2tau)',
        ),
        width=800, height=700,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()
