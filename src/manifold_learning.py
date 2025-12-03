
### Nicky 
# mainfold_approx.py

### VERY ROUGH NOTES ON WHAT TO DO HERE
# Function 2 (manifold learning): Takes in a bunch of points. Returns a tangential basis and a normal basis for each point (ideally this is orthonormal)
# - If KNN - Takes in a number of neigbours to use

# calculate_tangent_and_normal(dataset) -> norm_tangent_basis = dict[datapoint, [tangent_basis, normal_basis]] 
#     - This could be input to the diffusion model so that we generate these things on the fly


# Manifold learning thing
# - We need to take in a data set. And then we need to output some representation of a manifold.


# Return Structure of the manifold function (for now... this could be very inefficient. 
# So it could be improved later )
# dict = {
#     0: {
#         norm: [[1,1,1,1], [2,2,2,2]]
#         tangent: [[2,2,2,5,6]]
#     }
#     1: {
#         norm: [[1,1,1,1], [2,2,2,2]]
#         tangent: [[2,2,2,5,6]]
#     }
# }


from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LocalFrames:
    """
    Local tangent + normal frames estimated at each data point.

    X        : (N, D)
    tangent  : (N, D, d_t)
    normal   : (N, D, d_n)
    """
    X: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    intrinsic_dim: int


def estimate_tangent_normal_knn_pca(
    X: np.ndarray,
    intrinsic_dim: int,
    n_neighbors: int = 20,
    include_self: bool = True,
) -> LocalFrames:
    """
    Estimate tangent and normal spaces via KNN + local PCA.

    Parameters
    ----------
    X : (N, D)
        Data points in ambient space.
    intrinsic_dim : int
        Assumed intrinsic manifold dimension (d_t).
        For swiss roll / torus, this is 2.
    n_neighbors : int
        Number of neighbors for local PCA (including self if include_self=True).
    include_self : bool
        Whether to include the point itself in its neighborhood.

    Returns
    -------
    LocalFrames
        X        : (N, D)
        tangent  : (N, D, d_t)
        normal   : (N, D, d_n)
    """
    X = np.asarray(X)
    N, D = X.shape
    d_t = intrinsic_dim
    if d_t <= 0 or d_t >= D:
        raise ValueError(f"intrinsic_dim must be in [1, D-1], got {d_t} with D={D}")

    # --- Step 1: KNN neighborhoods (brute force for now) ---
    # Distance matrix: (N, N)
    # NOTE: this is O(N^2 D). Fine for moderate N; optimize later if needed.
    sq_norms = np.sum(X**2, axis=1, keepdims=True)  # (N, 1)
    dists_sq = sq_norms + sq_norms.T - 2 * (X @ X.T)
    dists_sq = np.maximum(dists_sq, 0.0)

    # Sort neighbors for each point
    # indices_sorted[i] gives neighbor indices sorted by distance
    indices_sorted = np.argsort(dists_sq, axis=1)

    if include_self:
        # first neighbor is self, so we just take the first n_neighbors
        k = min(n_neighbors, N)
        neighbor_indices = indices_sorted[:, :k]  # (N, k)
    else:
        # skip self (distance zero), take next n_neighbors
        k = min(n_neighbors, N - 1)
        neighbor_indices = indices_sorted[:, 1 : k + 1]  # (N, k)

    # --- Step 2: Local PCA for each point ---
    d_n = D - d_t

    tangent = np.zeros((N, D, d_t), dtype=float)
    normal = np.zeros((N, D, d_n), dtype=float)

    for i in range(N):
        neigh_idx = neighbor_indices[i]          # (k,)
        neigh = X[neigh_idx]                    # (k, D)

        # Center the neighborhood
        mean = np.mean(neigh, axis=0, keepdims=True)  # (1, D)
        neigh_centered = neigh - mean                 # (k, D)

        # SVD on centered neighborhood: (k x D)
        # Vt shape: (min(k, D), D); rows are principal directions in R^D.
        _, _, Vt = np.linalg.svd(neigh_centered, full_matrices=False)

        # Tangent = top d_t PCs
        # Normal  = remaining PCs
        # Ensure we have at least d_t components:
        if Vt.shape[0] < d_t:
            raise RuntimeError(
                f"Not enough singular vectors ({Vt.shape[0]}) for intrinsic_dim={d_t}"
            )

        V_tan = Vt[:d_t, :]            # (d_t, D)
        V_norm = Vt[d_t:, :]           # (d_n, D) where d_n = D - d_t

        # Store as (D, d_t) and (D, d_n) to match your T: (N, D, 2) convention
        tangent[i] = V_tan.T           # (D, d_t)
        if d_n > 0:
            normal[i] = V_norm.T       # (D, d_n)

    return LocalFrames(X=X, tangent=tangent, normal=normal, intrinsic_dim=d_t)
