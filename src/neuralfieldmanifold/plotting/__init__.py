"""Opt-in scientific plotting. Functions never display or save implicitly."""

import numpy as np


def embedding(points, *, ax=None, color="black", linewidth=0.6):
    """Plot a three-coordinate trajectory and return its Matplotlib axes."""
    import matplotlib.pyplot as plt

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected shape (N, 3)")
    if ax is None:
        ax = plt.figure().add_subplot(111, projection="3d")
    ax.plot(*points.T, color=color, lw=linewidth)
    ax.set(xlabel="x(t)", ylabel="x(t-tau)", zlabel="x(t-2tau)")
    return ax


def torus(points, fit, *, opacity=0.15):
    """Return a rotatable Plotly point cloud and fitted ring/torus surface."""
    import plotly.graph_objects as go

    points = np.asarray(points)
    theta = np.linspace(0, 2 * np.pi, 64)
    if fit.kind == "ring":
        radial = np.linspace(0, 1, 24)
        phi, weight = np.meshgrid(theta, radial)
        outer1, outer2 = fit.major_radii
        inner1, inner2 = fit.inner_radii
        u = (inner1 + weight * (outer1 - inner1)) * np.cos(phi)
        v = (inner2 + weight * (outer2 - inner2)) * np.sin(phi)
        surface = (
            fit.center[:, None, None]
            + fit.u_axis[:, None, None] * u
            + fit.v_axis[:, None, None] * v
        )
    else:
        phi, angle = np.meshgrid(theta, theta)
        r1, r2 = fit.major_radii
        denom = np.sqrt((r2 * np.cos(phi)) ** 2 + (r1 * np.sin(phi)) ** 2) + 1e-12
        outward = (
            r2 * np.cos(phi)[None, :, :] * fit.u_axis[:, None, None]
            + r1 * np.sin(phi)[None, :, :] * fit.v_axis[:, None, None]
        ) / denom
        surface = (
            fit.center[:, None, None]
            + r1 * np.cos(phi) * fit.u_axis[:, None, None]
            + r2 * np.sin(phi) * fit.v_axis[:, None, None]
            + fit.tube_radius * np.cos(angle) * outward
            + fit.tube_radius * np.sin(angle) * fit.normal[:, None, None]
        )
    fig = go.Figure(
        [
            go.Surface(
                x=surface[0],
                y=surface[1],
                z=surface[2],
                opacity=opacity,
                colorscale=[[0, "gray"], [1, "gray"]],
                showscale=False,
            ),
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color="black", width=2),
                name="Signal",
            ),
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="x(t)",
            yaxis_title="x(t-tau)",
            zaxis_title="x(t-2tau)",
        )
    )
    return fig
