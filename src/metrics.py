# src/metrics.py

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Tuple

from src.generate_data import generate_swiss_roll


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _to_torch(
    x: np.ndarray | torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a numpy array or torch tensor to a torch.Tensor on the given device.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Input data.
    device : torch.device or None
        Device to move the tensor to. If None, keep existing device (or CPU for numpy).
    dtype : torch.dtype
        Target dtype (default: float32).

    Returns
    -------
    torch.Tensor
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _subsample_points(
    X: torch.Tensor,
    max_points: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    (Optionally) random subsample rows of X to at most max_points.

    Parameters
    ----------
    X : (N, D) torch.Tensor
        Input point cloud.
    max_points : int
        Maximum number of points to keep. If N <= max_points, X is returned unchanged.
    device : torch.device or None
        Device for the random permutation (defaults to X.device).

    Returns
    -------
    X_sub : (M, D) torch.Tensor
        Subsampled point cloud (M <= max_points).
    """
    if device is None:
        device = X.device

    N = X.shape[0]
    if max_points is None or max_points <= 0 or N <= max_points:
        return X

    idx = torch.randperm(N, device=device)[:max_points]
    return X[idx]


# ---------------------------------------------------------------------------
# Generic point-cloud metrics
# ---------------------------------------------------------------------------

def one_sided_chamfer(
    X_source: np.ndarray | torch.Tensor,
    X_target: np.ndarray | torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    max_source: int = 2000,
    max_target: int = 2000,
) -> float:
    """
    One-sided Chamfer-style distance: for each point in X_source, find the
    squared distance to the closest point in X_target, then average.

    This is useful when X_target is considered the "reference" set
    (e.g., a dense manifold approximation), and we care about how well
    X_source lies near it.

    Parameters
    ----------
    X_source : (N_s, D) np.ndarray or torch.Tensor
        Source point cloud (e.g., generated samples).
    X_target : (N_t, D) np.ndarray or torch.Tensor
        Target point cloud (e.g., training data or manifold samples).
    device : torch.device or None
        Device to run the computation on. If None, choose CUDA if available.
    max_source : int
        Optionally subsample the source points to at most this many points.
    max_target : int
        Optionally subsample the target points to at most this many points.

    Returns
    -------
    float
        Mean squared distance from each source point to its closest target point.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xs = _to_torch(X_source, device=device)  # (N_s, D)
    Xt = _to_torch(X_target, device=device)  # (N_t, D)

    Xs = _subsample_points(Xs, max_source, device=device)
    Xt = _subsample_points(Xt, max_target, device=device)

    # (N_s, N_t) pairwise distances
    dists = torch.cdist(Xs, Xt)      # L2 distances
    dists_sq = dists ** 2

    # For each source point, distance to closest target
    min_dist_sq, _ = torch.min(dists_sq, dim=1)  # (N_s,)

    return torch.mean(min_dist_sq).item()


def chamfer_distance(
    X_true: np.ndarray | torch.Tensor,
    X_gen: np.ndarray | torch.Tensor,
    *,
    device: Optional[torch.device] = None,
    max_true: int = 2000,
    max_gen: int = 2000,
) -> float:
    """
    Symmetric Chamfer distance between two point clouds.

    Chamfer(X_true, X_gen) =
        E_{x_gen in X_gen} [min_{x_true in X_true} ||x_gen - x_true||^2]
      + E_{x_true in X_true} [min_{x_gen in X_gen} ||x_true - x_gen||^2]

    This is your main metric when the underlying manifold is unknown
    and you only have samples (e.g., dataset vs generated).

    Parameters
    ----------
    X_true : (N_true, D) np.ndarray or torch.Tensor
        Reference / "ground-truth" sample cloud (e.g., training data).
    X_gen : (N_gen, D) np.ndarray or torch.Tensor
        Generated sample cloud.
    device : torch.device or None
        Device on which to run the computation.
    max_true : int
        Optionally subsample true points to at most this many points.
    max_gen : int
        Optionally subsample generated points to at most this many points.

    Returns
    -------
    float
        Symmetric Chamfer distance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_true_t = _to_torch(X_true, device=device)
    X_gen_t = _to_torch(X_gen, device=device)

    X_true_t = _subsample_points(X_true_t, max_true, device=device)
    X_gen_t = _subsample_points(X_gen_t, max_gen, device=device)

    # (N_gen, N_true)
    dists = torch.cdist(X_gen_t, X_true_t)
    dists_sq = dists ** 2

    # gen -> true
    min_dist_gen_to_true, _ = torch.min(dists_sq, dim=1)  # (N_gen,)
    # true -> gen
    min_dist_true_to_gen, _ = torch.min(dists_sq, dim=0)  # (N_true,)

    chamfer = torch.mean(min_dist_gen_to_true) + torch.mean(min_dist_true_to_gen)
    return chamfer.item()


# ---------------------------------------------------------------------------
# Metrics for known swiss-roll manifold
# ---------------------------------------------------------------------------

def _build_swiss_roll_manifold_grid(
    embed_dim: int,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_u: int,
    n_v: int,
    E: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Build a dense, noise-free sampling of the swiss roll manifold.

    We approximate the manifold by sampling (u, v) on a regular grid and
    mapping via generate_swiss_roll (which also handles arbitrary embedding
    dimension and linear embedding E).

    Parameters
    ----------
    embed_dim : int
        Ambient dimension D.
    u_range, v_range : (float, float)
        Parameter ranges for u and v.
    n_u, n_v : int
        Grid resolution along u and v.
    E : np.ndarray or None
        Optional embedding matrix of shape (D, 3). If None, generate_swiss_roll
        chooses an appropriate embedding (identity if D=3, random orthonormal otherwise).
    random_state : int
        Seed for reproducibility (if E is None and embed_dim > 3).

    Returns
    -------
    X_manifold : (n_u * n_v, D) np.ndarray
        Noise-free manifold sampling.
    """
    # Sample (u, v) on a grid
    u_min, u_max = u_range
    v_min, v_max = v_range

    u_lin = np.linspace(u_min, u_max, n_u)
    v_lin = np.linspace(v_min, v_max, n_v)
    uu, vv = np.meshgrid(u_lin, v_lin, indexing="ij")  # (n_u, n_v)

    u_flat = uu.reshape(-1)  # (n_u * n_v,)
    v_flat = vv.reshape(-1)

    # generate_swiss_roll ignores provided u/v and samples internally,
    # so we implement a small custom version here using the same formulae,
    # but we can piggyback on generate_swiss_roll to get a consistent embedding E.

    # 1) Base swiss roll in R^3
    x = u_flat * np.cos(u_flat)
    y = v_flat
    z = u_flat * np.sin(u_flat)
    X3 = np.stack([x, y, z], axis=-1)  # (N, 3)

    # 2) Embedding matrix
    if E is not None:
        E_used = np.asarray(E)
        if E_used.shape != (embed_dim, 3):
            raise ValueError(
                f"E must have shape (embed_dim, 3) = ({embed_dim}, 3), got {E_used.shape}"
            )
    else:
        # Use generate_swiss_roll just to get a consistent E for the given embed_dim
        _, _, _, E_used = generate_swiss_roll(
            n_samples=1,
            embed_dim=embed_dim,
            u_range=u_range,
            v_range=v_range,
            noise_sigma=0.0,
            random_state=random_state,
            embed_matrix=None,
        )

    # 3) Embed
    X_manifold = X3 @ E_used.T  # (N, D)
    return X_manifold.astype(np.float32)


def swiss_roll_manifold_distance(
    X_gen: np.ndarray | torch.Tensor,
    *,
    embed_dim: int = 3,
    u_range: Tuple[float, float] = (1.5 * np.pi, 4.5 * np.pi),
    v_range: Tuple[float, float] = (0.0, 10.0),
    n_u: int = 200,
    n_v: int = 50,
    E: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    max_gen: int = 2000,
) -> float:
    """
    Mean squared distance from generated samples to the true swiss-roll manifold.

    We approximate the analytic manifold by a dense, noise-free sampling
    of the swiss roll parameterization (u, v) mapped to R^D via the same
    embedding matrix E (or a consistent one if E is None).

    This is essentially a one-sided Chamfer distance:
        E_{x_gen} [min_{x_manifold} ||x_gen - x_manifold||^2]

    Parameters
    ----------
    X_gen : (N_gen, D) np.ndarray or torch.Tensor
        Generated samples in R^D. Must satisfy D == embed_dim.
    embed_dim : int
        Ambient dimension D.
    u_range, v_range : (float, float)
        Ranges for u and v used in the swiss roll generation.
    n_u, n_v : int
        Grid resolution for approximating the manifold.
    E : np.ndarray or None
        Embedding matrix (D, 3). If None, we construct a consistent one.
    device : torch.device or None
        Device to run the computation on.
    max_gen : int
        Optionally subsample generated points for speed.

    Returns
    -------
    float
        Mean squared distance from each generated point to the closest
        point on the approximated swiss roll manifold.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_gen_t = _to_torch(X_gen, device=device)  # (N_gen, D)
    N_gen, D = X_gen_t.shape
    if D != embed_dim:
        raise ValueError(f"X_gen has dim {D}, expected embed_dim={embed_dim}")

    # Subsample generated points if needed
    X_gen_t = _subsample_points(X_gen_t, max_gen, device=device)

    # Build dense manifold sampling in numpy, then convert to torch
    X_manifold_np = _build_swiss_roll_manifold_grid(
        embed_dim=embed_dim,
        u_range=u_range,
        v_range=v_range,
        n_u=n_u,
        n_v=n_v,
        E=E,
    )  # (N_manifold, D)
    X_manifold_t = _to_torch(X_manifold_np, device=device)

    # Compute one-sided distance: gen -> manifold
    dists = torch.cdist(X_gen_t, X_manifold_t)  # (N_gen, N_manifold)
    dists_sq = dists ** 2
    min_dist_sq, _ = torch.min(dists_sq, dim=1)  # (N_gen,)

    return torch.mean(min_dist_sq).item()
