import numpy as np
from scipy import optimize
from plotly import graph_objects as go


def two_torus_fit(points, lam=1.0, hole_ratio=0.75, n_modes=3):
    """
    Fit an elliptical torus to a 3-D point cloud.

    Objective
    ─────────
    COVERAGE    : Σ max(0, dᵢ − r)²   — contain the point cloud
    COMPACTNESS : don't exceed PCA extents + protect the hole

    lam   : tradeoff (0 = pure containment, ∞ = locked to PCA)
    hole_ratio : tube radius penalised when r > hole_ratio * min(R1,R2)
                 0.75 is a good default — allows a thick tube but keeps
                 a clear hole even with outliers cutting through it
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

    r_tgt  = pc_ext[2] / 2.0
    R1_tgt = max(pc_ext[0] / 2.0 - r_tgt, r_tgt * 0.5)
    R2_tgt = max(pc_ext[1] / 2.0 - r_tgt, r_tgt * 0.5)

    u0, v0, n0 = Vt[0], Vt[1], Vt[2]

    # ── helpers ──────────────────────────────────────────────────────
    def _frame(dn):
        R1   = np.linalg.norm(dn) + 1e-12
        n_ax = dn / R1
        u    = u0 - np.dot(u0, n_ax) * n_ax
        u_ax = u / (np.linalg.norm(u) + 1e-12)
        v_ax = np.cross(n_ax, u_ax)
        v_ax = v_ax / (np.linalg.norm(v_ax) + 1e-12)
        return R1, n_ax, u_ax, v_ax

    def _nearest_phi(cu, cv, R1, R2, iters=4):
        phi = np.arctan2(cv * R1, cu * R2)
        for _ in range(iters):
            cp, sp = np.cos(phi), np.sin(phi)
            dx, dy = cu - R1 * cp, cv - R2 * sp
            g = dx * R1 * sp - dy * R2 * cp
            h = dx * R1 * cp + dy * R2 * sp + (R1*sp)**2 + (R2*cp)**2
            phi -= np.clip(g / (h + 1e-12), -0.5, 0.5)
        return phi

    def _dist(c, n_ax, u_ax, v_ax, R1, R2, pts):
        dp   = pts - c
        h    = dp @ n_ax
        perp = dp - h[:, None] * n_ax
        cu, cv = perp @ u_ax, perp @ v_ax
        phi  = _nearest_phi(cu, cv, R1, R2)
        near = ((R1*np.cos(phi))[:, None] * u_ax
                + (R2*np.sin(phi))[:, None] * v_ax + c)
        return np.linalg.norm(pts - near, axis=1), phi

    # ── residuals ────────────────────────────────────────────────────
    def residuals(params):
        c  = params[:3]
        dn = params[3:6]
        R1, n_ax, u_ax, v_ax = _frame(dn)
        R2 = R1 * np.exp(params[6])
        r  = np.exp(params[7])

        d, phi = _dist(c, n_ax, u_ax, v_ax, R1, R2, pts)

        # COVERAGE: only outside points
        coverage = np.maximum(0.0, d - r)

        # COMPACTNESS: don't exceed PCA extents + protect the hole
        w = lam * np.sqrt(N)
        Rmin = min(R1, R2)
        hole_overshoot = max(0.0, r - hole_ratio * Rmin) / (Rmin + 1e-12)

        compact = np.array([
            w * max(0.0, 2*(R1+r) - pc_ext[0]) / (pc_ext[0] + 1e-12),
            w * max(0.0, 2*(R2+r) - pc_ext[1]) / (pc_ext[1] + 1e-12),
            w * max(0.0, 2*r      - pc_ext[2]) / (pc_ext[2] + 1e-12),
            # hole penalty: 3× weight because losing the hole is
            # geometrically catastrophic (torus → blob)
            3.0 * w * hole_overshoot,
        ])

        # HOMOGENEITY: keep backbone centred
        lam_h = 0.3
        raw   = d - r
        homog = []
        for k in range(1, n_modes + 1):
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.cos(k*phi)))
            homog.append(lam_h * np.sqrt(N) * np.mean(raw * np.sin(k*phi)))

        return np.concatenate([coverage, compact, np.array(homog)])

    # ── init & bounds ────────────────────────────────────────────────
    ratio0 = np.clip(R2_tgt / (R1_tgt + 1e-12), 0.1, 1.0)
    x0 = np.concatenate([
        center0,
        n0 * R1_tgt,
        [np.log(ratio0)],
        [np.log(max(r_tgt, 1e-3))],
    ])

    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0,  np.inf)
    max_R1 = pc_range[0] / 2.0
    lb[3:6], ub[3:6] = -max_R1, max_R1
    lb[6] = np.log(0.05)
    ub[6] = np.log(np.clip(pc_range[1]/(pc_range[0]+1e-12), 0.05, 1.0))
    lb[7] = np.log(1e-3)
    ub[7] = np.log(pc_range[2] / 2.0 + 1e-6)

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
    R1, n_ax, u_ax, v_ax = _frame(dn)
    R2 = R1 * np.exp(result.x[6])
    r  = np.exp(result.x[7])

    if abs(np.dot(u_ax, u0)) < abs(np.dot(v_ax, u0)):
        R1, R2 = R2, R1
        u_ax, v_ax = v_ax, u_ax

    d, _ = _dist(c, n_ax, u_ax, v_ax, R1, R2, pts)
    signed       = d - r
    outside_dist = np.maximum(0.0, signed)
    inside_mask  = signed <= 0

    dp   = pts - c
    perp = dp - (dp @ n_ax)[:, None] * n_ax
    ell  = (perp @ u_ax/(R1+1e-12))**2 + (perp @ v_ax/(R2+1e-12))**2
    in_hole = (ell < 1.0) & (~inside_mask)

    # R² using residuals where inside-volume points have 0 residual
    ss_res = np.sum(outside_dist**2)
    ss_tot = np.sum(np.sum((pts - pts.mean(axis=0))**2, axis=1))
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

    return dict(
        center=c, direction=n_ax, u_axis=u_ax, v_axis=v_ax,
        R1=float(R1), R2=float(R2), minor_radius=float(r),
        hole_quality=float(1.0 - r / (min(R1, R2) + 1e-12)),
        r_squared=float(r_squared),
        mse=float(np.mean(outside_dist**2)),
        mean_error=float(np.mean(outside_dist)),
        frac_inside=float(inside_mask.sum() / N),
        frac_in_hole=float(in_hole.sum() / N),
        pc_extents=pc_ext,
        pc_ranges=pc_range,
    )

def plot_two_torus_fit(entry, n_surf=60, alpha_surf=0.15):

    pts    = np.asarray(entry['points'])
    center = np.asarray(entry['center'])
    axis   = np.asarray(entry['direction'])
    u_ax   = np.asarray(entry['u_axis'])
    v_ax   = np.asarray(entry['v_axis'])
    R1     = float(entry['R1'])
    R2     = float(entry['R2'])
    r      = float(entry['minor_radius'])

    # classify inside / outside
    dp     = pts - center
    along  = np.outer(dp @ axis, axis)
    perp   = dp - along
    cu     = perp @ u_ax
    cv     = perp @ v_ax
    theta  = np.arctan2(cv / (R2 + 1e-12), cu / (R1 + 1e-12))
    ex     = R1 * np.cos(theta)
    ey     = R2 * np.sin(theta)
    nearest = ex[:, None] * u_ax + ey[:, None] * v_ax + center
    d      = np.linalg.norm(pts - nearest, axis=1)
    inside = d <= r

    # elliptical torus mesh — use true ellipse outward normal
    phi   = np.linspace(0, 2 * np.pi, n_surf)
    theta_m = np.linspace(0, 2 * np.pi, n_surf)
    PHI, THETA = np.meshgrid(phi, theta_m)
    bx = R1 * np.cos(PHI)
    by = R2 * np.sin(PHI)
    denom = np.sqrt((R2 * np.cos(PHI))**2 + (R1 * np.sin(PHI))**2) + 1e-12
    out_u = R2 * np.cos(PHI) / denom
    out_v = R1 * np.sin(PHI) / denom
    X = center[0] + bx*u_ax[0] + by*v_ax[0] + r*np.cos(THETA)*(out_u*u_ax[0]+out_v*v_ax[0]) + r*np.sin(THETA)*axis[0]
    Y = center[1] + bx*u_ax[1] + by*v_ax[1] + r*np.cos(THETA)*(out_u*u_ax[1]+out_v*v_ax[1]) + r*np.sin(THETA)*axis[1]
    Z = center[2] + bx*u_ax[2] + by*v_ax[2] + r*np.cos(THETA)*(out_u*u_ax[2]+out_v*v_ax[2]) + r*np.sin(THETA)*axis[2]

    fig = go.Figure()

    # torus surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, opacity=alpha_surf,
        colorscale=[[0, 'steelblue'], [1, 'steelblue']],
        showscale=False, name='Torus',
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
            grp = 'inside'
        else:
            show = first_out; first_out = False
            name = f'outside ({n_out})'
            grp = 'outside'
        fig.add_trace(go.Scatter3d(
            x=seg[:, 0], y=seg[:, 1], z=seg[:, 2],
            mode='lines', line=dict(color=color, width=3),
            name=name, showlegend=show, legendgroup=grp,
        ))
        i = j

    t_start, t_end = entry['start_sec'], entry['end_sec']
    fig.update_layout(
        title=dict(text=(
            f"Elliptical torus {t_start:.1f}–{t_end:.1f} s | "
            f"R1={R1:.1f}  R2={R2:.1f}  r={r:.1f}  MSE={entry['mse']:.2f}"
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