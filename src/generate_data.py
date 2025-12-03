# src/data/generate_data.py

from __future__ import annotations

from typing import Dict, Tuple, Optional, Literal
import numpy as np


DatasetName = Literal["swiss_roll", "torus"]


def generate_swiss_roll(
    n_samples: int,
    embed_dim: int = 3,
    u_range: Tuple[float, float] = (1.5 * np.pi, 4.5 * np.pi),
    v_range: Tuple[float, float] = (0.0, 10.0),
    noise_sigma: float = 0.0,
    random_state: Optional[int] = None,
    embed_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D swiss roll surface embedded in arbitrary ambient dimension.

    Base swiss roll (in 3D):
        x = u * cos(u)
        y = v
        z = u * sin(u)

    Then embed R^3 -> R^D via a linear map E (D x 3):
      - If embed_dim == 3 and embed_matrix is None: E = I_3 (no change).
      - If embed_matrix is provided: use it (must be shape (embed_dim, 3)).
      - Else: draw a random orthonormal (embed_dim x 3) matrix and use that.

    Parameters
    ----------
    n_samples : int
        Number of points to sample on the swiss roll.
    embed_dim : int
        Ambient dimension D >= 3.
    u_range : (float, float)
        Range for u parameter (controls how many turns).
    v_range : (float, float)
        Range for v parameter (height / width).
    noise_sigma : float
        Std dev of isotropic Gaussian noise added in the *base 3D* space.
    random_state : int or None
        Seed for reproducibility.
    embed_matrix : np.ndarray or None
        Optional embedding matrix E of shape (embed_dim, 3).

    Returns
    -------
    X : (n_samples, embed_dim)
        Points on the swiss roll embedded in R^embed_dim.
    u : (n_samples,)
        u parameters for each point.
    v : (n_samples,)
        v parameters for each point.
    E : (embed_dim, 3)
        The embedding matrix actually used.
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
                f"embed_matrix must have shape ({embed_dim}, 3), got {E.shape}"
            )
    else:
        if embed_dim == 3:
            E = np.eye(3)
        else:
            # Random orthonormal embed_dim x 3 matrix via QR
            A = rng.standard_normal(size=(embed_dim, 3))
            Q, _ = np.linalg.qr(A)
            E = Q[:, :3]  # (embed_dim, 3), columns are orthonormal

    # Embed: X = X3 @ E^T   (n_samples x 3) * (3 x embed_dim) = (n_samples x embed_dim)
    X = X3 @ E.T

    return X, u, v, E


def generate_torus(
    n_samples: int,
    embed_dim: int = 3,
    R: float = 2.0,  # major radius
    r: float = 1.0,  # minor radius
    noise_sigma: float = 0.0,
    random_state: Optional[int] = None,
    embed_matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points on a 2D torus embedded in R^embed_dim.

    Base torus in R^3 (theta around hole, phi around tube):
        x = (R + r cos(phi)) cos(theta)
        y = (R + r cos(phi)) sin(theta)
        z = r sin(phi)

    Parameters
    ----------
    n_samples : int
        Number of points to sample.
    embed_dim : int
        Ambient dimension D >= 3.
    R : float
        Major radius.
    r : float
        Minor radius.
    noise_sigma : float
        Std dev of isotropic Gaussian noise in base 3D space.
    random_state : int or None
        Seed for reproducibility.
    embed_matrix : np.ndarray or None
        Embedding matrix E of shape (embed_dim, 3), optional.

    Returns
    -------
    X : (n_samples, embed_dim)
        Points on the torus embedded in R^embed_dim.
    theta : (n_samples,)
        Angle around the main circle.
    phi : (n_samples,)
        Angle around the tube.
    """
    if embed_dim < 3:
        raise ValueError("embed_dim must be >= 3 for a torus surface.")

    rng = np.random.default_rng(random_state)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)  # angle around hole
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)  # angle around tube

    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    X3 = np.stack([x, y, z], axis=-1)

    if noise_sigma > 0.0:
        X3 = X3 + rng.normal(scale=noise_sigma, size=X3.shape)

    # Build embedding matrix (reuse swiss-roll logic)
    if embed_matrix is not None:
        E = np.asarray(embed_matrix)
        if E.shape != (embed_dim, 3):
            raise ValueError(
                f"embed_matrix must have shape ({embed_dim}, 3), got {E.shape}"
            )
    else:
        if embed_dim == 3:
            E = np.eye(3)
        else:
            A = rng.standard_normal(size=(embed_dim, 3))
            Q, _ = np.linalg.qr(A)
            E = Q[:, :3]

    X = X3 @ E.T
    return X, theta, phi


def generate_dataset(
    name: DatasetName,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Generic entry-point to generate synthetic datasets.

    Parameters
    ----------
    name : {"swiss_roll", "torus"}
        Which dataset to create.
    kwargs : dict
        Passed through to the underlying function.

    Returns
    -------
    A dict containing at minimum:
      - "X": data matrix of shape (n_samples, d)
    And optionally:
      - "u", "v", "E" for swiss_roll
      - "theta", "phi" for torus
    """
    if name == "swiss_roll":
        X, u, v, E = generate_swiss_roll(**kwargs)
        return {"X": X, "u": u, "v": v, "E": E}

    if name == "torus":
        X, theta, phi = generate_torus(**kwargs)
        return {"X": X, "theta": theta, "phi": phi}

    raise ValueError(f"Unknown dataset name: {name!r}")
