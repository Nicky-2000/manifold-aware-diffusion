
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Sequence

from src.manifold_learning import LocalFrames

def plot_swiss_roll_3d(
    X: np.ndarray,
    color: np.ndarray | None = None,
    title: str = "Swiss Roll in 3D",
):
    """
    Plot a 2D swiss roll embedded in 3D using Plotly.

    Parameters
    ----------
    X : (N, 3)
        Swiss roll points in 3D.
    color : (N,) or None
        Optional color values (e.g., u parameter). If None, uses z-values.
    title : str
        Plot title.
    """
    X = np.asarray(X)
    if X.shape[1] != 3:
        raise ValueError(f"X must have shape (N, 3) for 3D plotting, got {X.shape}")

    if color is None:
        color = X[:, 2]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=color,
                    colorscale="Viridis",
                    opacity=0.8,
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        template="plotly_white",
    )

    return fig




def plot_swiss_with_local_frames_3d(
    frames: LocalFrames,
    indices: Optional[Sequence[int]] = None,
    every_n: Optional[int] = None,
    plane_scale: float = 0.5,
    normal_scale: float = 0.7,
    title: str = "Swiss Roll with Tangent Planes and Normals",
):
    """
    Plot swiss roll points and overlay tangent planes + normal vectors
    at a subset of points using LocalFrames.

    Parameters
    ----------
    frames : LocalFrames
        Contains:
          - X       : (N, 3)   data points in R^3
          - tangent : (N, 3, d_t)  tangent basis (columns per point)
          - normal  : (N, 3, d_n)  normal basis (columns per point)
    indices : list[int] or np.ndarray, optional
        Explicit indices where to draw frames.
    every_n : int, optional
        If indices is None, draw frames every_n points along X.
    plane_scale : float
        Half-size of the tangent plane patch around each point.
    normal_scale : float
        Length of the normal vector drawn at each point.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    X = np.asarray(frames.X)
    tangent = np.asarray(frames.tangent)
    normal = np.asarray(frames.normal)

    N, D = X.shape
    if D != 3:
        raise ValueError(
            f"plot_swiss_with_local_frames_3d expects X in R^3, got shape {X.shape}"
        )

    if tangent.shape[0] != N or tangent.shape[1] != 3:
        raise ValueError(
            f"tangent must have shape (N, 3, d_t), got {tangent.shape}"
        )

    d_t = tangent.shape[2]
    if d_t < 2:
        raise ValueError(
            f"Need at least 2 tangent directions to form a plane, got d_t={d_t}"
        )

    d_n = normal.shape[2] if normal.ndim == 3 else 0

    # Decide which points get frames
    if indices is None:
        if every_n is None:
            every_n = max(N // 20, 1)
        indices = np.arange(0, N, every_n)
    else:
        indices = np.asarray(indices)

    fig = go.Figure()

    # Base point cloud
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2],
            mode="markers",
            marker=dict(size=2, color=X[:, 2], colorscale="Viridis", opacity=0.5),
            name="Swiss roll",
        )
    )

    first_plane = True
    first_norm = True

    for idx in indices:
        p = X[idx]                 # (3,)

        # Use first two tangent vectors for the plane
        t1 = tangent[idx, :, 0]    # (3,)
        t2 = tangent[idx, :, 1]    # (3,)

        # Four corners of a small parallelogram in the tangent plane
        c00 = p - plane_scale * t1 - plane_scale * t2
        c01 = p - plane_scale * t1 + plane_scale * t2
        c10 = p + plane_scale * t1 - plane_scale * t2
        c11 = p + plane_scale * t1 + plane_scale * t2

        # Build a tiny quad as two triangles via Mesh3d
        xs = [c00[0], c01[0], c10[0], c11[0]]
        ys = [c00[1], c01[1], c10[1], c11[1]]
        zs = [c00[2], c01[2], c10[2], c11[2]]

        fig.add_trace(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                # Two triangles: (0,1,2) and (1,2,3)
                i=[0, 1],
                j=[1, 2],
                k=[2, 3],
                opacity=0.4,
                color="blue",
                name="Tangent plane" if first_plane else None,
                showlegend=first_plane,
            )
        )
        first_plane = False

        # Normal vector: use first normal direction if available
        if d_n > 0:
            n_vec = normal[idx, :, 0]          # (3,)
        else:
            # Fallback: cross product of the two tangent vectors
            n_vec = np.cross(t1, t2)
            n_norm = np.linalg.norm(n_vec)
            if n_norm > 1e-12:
                n_vec = n_vec / n_norm

        p_n_end = p + normal_scale * n_vec

        fig.add_trace(
            go.Scatter3d(
                x=[p[0], p_n_end[0]],
                y=[p[1], p_n_end[1]],
                z=[p[2], p_n_end[2]],
                mode="lines",
                name="Normal" if first_norm else None,
                showlegend=first_norm,
                line=dict(width=4, dash="dot"),
            )
        )
        first_norm = False

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        template="plotly_white",
    )

    return fig
