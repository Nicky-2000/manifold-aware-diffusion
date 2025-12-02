

# Generate_Dataset(some parameters) -> dataset - matrix

### Nicky
# generate_data.py
# Funciton 1: Will generate a 2d swiss roll in arbitrary dimensions  -> Returns a bunch of points. 
# - takes in a number of samples to generate

# - Swiss Roll 
# - Torus
# - Non synthetic - Load in from some file? 
# - Images... Dogs?

# src/data/loaders.py

import numpy as np

def generate_swiss_roll(
    n_samples,
    embed_dim=3,
    u_range=(1.5 * np.pi, 4.5 * np.pi),
    v_range=(0.0, 10.0),
    noise_sigma=0.0,
    random_state=None,
    embed_matrix=None,
):
    """
    Generate a 2D swiss roll surface embedded in arbitrary ambient dimension.

    Base swiss roll (in 3D):
        x = u * cos(u)
        y = v
        z = u * sin(u)

    Then we embed R^3 -> R^D via a linear map E (D x 3):
      - If embed_dim == 3 and embed_matrix is None: E = I_3 (no change).
      - If embed_matrix is provided: use it (must be shape (embed_dim, 3)).
      - Else: draw a random orthonormal D x 3 matrix and use that.

    Parameters
    ----------
    n_samples : int
        Number of points to sample on the swiss roll.
    embed_dim : int
        Ambient dimension D >= 3.
    u_range : tuple(float, float)
        Range for u parameter (controls how many turns).
    v_range : tuple(float, float)
        Range for v parameter (height / width).
    noise_sigma : float
        Std dev of isotropic Gaussian noise added in the *base 3D* space.
    random_state : int or None
        Seed for reproducibility.
    embed_matrix : np.ndarray or None
        Optional embedding matrix E of shape (embed_dim, 3). If provided,
        it is used directly. You can reuse this for tangents later.

    Returns
    -------
    X : np.ndarray, shape (n_samples, embed_dim)
        Points on the swiss roll embedded in R^embed_dim.
    u : np.ndarray, shape (n_samples,)
        u parameters for each point.
    v : np.ndarray, shape (n_samples,)
        v parameters for each point.
    E : np.ndarray, shape (embed_dim, 3)
        The embedding matrix actually used (so you can apply it to tangents).
    """
    if embed_dim < 3:
        raise ValueError("embed_dim must be >= 3 for a 2D swiss roll surface.")

    rng = np.random.default_rng(random_state)

    u_min, u_max = u_range
    v_min, v_max = v_range

    # Sample parameters
    u = rng.uniform(u_min, u_max, size=n_samples)
    v = rng.uniform(v_min, v_max, size=n_samples)

    # Base 3D swiss roll
    x = u * np.cos(u)
    y = v
    z = u * np.sin(u)
    X3 = np.stack([x, y, z], axis=-1)  # (n_samples, 3)

    # Optional noise in base 3D space
    if noise_sigma > 0.0:
        X3 = X3 + rng.normal(scale=noise_sigma, size=X3.shape)

    # Build embedding matrix E: R^3 -> R^embed_dim
    if embed_matrix is not None:
        E = np.asarray(embed_matrix)
        if E.shape != (embed_dim, 3):
            raise ValueError(
                f"embed_matrix must have shape (embed_dim, 3) = ({embed_dim}, 3), "
                f"got {E.shape}"
            )
    else:
        if embed_dim == 3:
            E = np.eye(3)
        else:
            # Random orthonormal D x 3 matrix via QR
            A = rng.standard_normal(size=(embed_dim, 3))
            Q, _ = np.linalg.qr(A)
            E = Q[:, :3]  # (embed_dim, 3), columns are orthonormal

    # Embed: X = X3 @ E^T   (n_samples x 3) * (3 x embed_dim) = (n_samples x embed_dim)
    X = X3 @ E.T

    return X, u, v, E


import plotly.graph_objects as go
import numpy as np

def plot_swiss_roll_3d(X, u=None, title="Swiss Roll in 3D"):
    """
    Plot a 2D swiss roll embedded in 3D using Plotly.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Swiss roll points in 3D.
    u : np.ndarray or None
        Optional color parameter (e.g., u parameter). If None, uses z-values.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    X = np.asarray(X)
    assert X.shape[1] == 3, "X must have shape (N, 3) for 3D plotting."

    if u is None:
        colors = X[:, 2]  # color by height as fallback
    else:
        colors = u

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors,
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
        template="plotly_white"
    )

    return fig

import numpy as np
