
import numpy as np

def compute_true_tangent_normal_swiss(u, v, E=None):
    """
    Compute true tangent basis (2D) and a canonical normal vector
    for the swiss roll at each (u, v), optionally embedded in R^D via E.

    Base swiss roll in R^3:
        x = u cos u
        y = v
        z = u sin u

    Tangent basis in R^3 is given by columns:
        du = d gamma / du
        dv = d gamma / dv

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        u parameters.
    v : np.ndarray, shape (N,)
        v parameters (not used in derivatives, but kept for consistency).
    E : np.ndarray or None
        Optional embedding matrix of shape (D, 3).
        If None, we stay in R^3.

    Returns
    -------
    T : np.ndarray, shape (N, D, 2)
        Tangent basis at each point (two orthonormal vectors in R^D).
    N_vec : np.ndarray, shape (N, D)
        Canonical unit normal vector at each point in R^D.
        In D=3, this spans the full normal space; in D>3, it's one
        distinguished normal direction.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    assert u.shape == v.shape, "u and v must have same shape"
    N_pts = u.shape[0]

    # ---- 3D partial derivatives ----
    du_x = np.cos(u) - u * np.sin(u)
    du_y = np.zeros_like(u)
    du_z = np.sin(u) + u * np.cos(u)

    dv_x = np.zeros_like(u)
    dv_y = np.ones_like(u)
    dv_z = np.zeros_like(u)

    du = np.stack([du_x, du_y, du_z], axis=-1)  # (N, 3)
    dv = np.stack([dv_x, dv_y, dv_z], axis=-1)  # (N, 3)

    # Stack into tangent basis in R^3: (N, 3, 2)
    T3 = np.stack([du, dv], axis=-1)

    # Orthonormalize columns (they're already orthogonal for this param,
    # but we normalize to be safe)
    for j in range(2):
        norms = np.linalg.norm(T3[:, :, j], axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        T3[:, :, j] = T3[:, :, j] / norms

    # Canonical normal in 3D via cross product of tangent basis vectors
    n3 = np.cross(T3[:, :, 0], T3[:, :, 1], axis=1)  # (N, 3)
    norms_n = np.linalg.norm(n3, axis=1, keepdims=True)
    norms_n = np.clip(norms_n, 1e-12, None)
    n3 = n3 / norms_n

    # ---- Embed to R^D if E is provided ----
    if E is None:
        # Stay in 3D
        D = 3
        T = T3  # (N, 3, 2)
        N_vec = n3  # (N, 3)
    else:
        E = np.asarray(E)
        D, three = E.shape
        assert three == 3, "E must have shape (D, 3)"
        # T[i] = E @ T3[i]   => (D,3)@(3,2) = (D,2)
        T = np.einsum("dj,njk->ndk", E, T3)  # (N, D, 2)
        # n[i] = E @ n3[i]
        N_vec = (E @ n3.T).T  # (N, D)

        # Re-normalize after embedding (numerical safety)
        for j in range(2):
            norms = np.linalg.norm(T[:, :, j], axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            T[:, :, j] = T[:, :, j] / norms

        norms_n = np.linalg.norm(N_vec, axis=1, keepdims=True)
        norms_n = np.clip(norms_n, 1e-12, None)
        N_vec = N_vec / norms_n

    return T, N_vec

def plot_swiss_with_frames_3d(
    X,
    T,
    N_vec,
    indices=None,
    every_n=None,
    scale_tangent=1.0,
    scale_normal=1.0,
    title="Swiss Roll with Tangent Planes and Normals",
):
    """
    Plot swiss roll points and overlay tangent basis vectors (2D plane)
    and a normal vector at selected points.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Swiss roll points in 3D.
    T : np.ndarray, shape (N, 3, 2)
        Tangent basis at each point (two orthonormal vectors).
    N_vec : np.ndarray, shape (N, 3)
        Canonical unit normal vector at each point.
    indices : list[int] or np.ndarray, optional
        Explicit indices where to draw frames.
    every_n : int, optional
        If indices is None, draw frames every_n points.
    scale_tangent : float
        Length scale for tangent vectors.
    scale_normal : float
        Length scale for normal vectors.
    title : str
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    X = np.asarray(X)
    T = np.asarray(T)
    N_vec = np.asarray(N_vec)

    N_pts, D = X.shape
    assert D == 3, "This plotting helper assumes X is in R^3."
    assert T.shape == (N_pts, 3, 2), "T must be (N, 3, 2)"
    assert N_vec.shape == (N_pts, 3), "N_vec must be (N, 3)"

    # Decide which indices to use
    if indices is None:
        if every_n is None:
            every_n = max(N_pts // 20, 1)
        indices = np.arange(0, N_pts, every_n)
    else:
        indices = np.asarray(indices)

    fig = go.Figure()

    # Base point cloud
    fig.add_trace(go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        mode="markers",
        marker=dict(size=2, color=X[:, 2], colorscale="Viridis", opacity=0.5),
        name="Swiss roll",
    ))

    first_tan = True
    first_norm = True

    for idx in indices:
        p = X[idx]              # point in R^3
        t1 = T[idx, :, 0]       # first tangent dir
        t2 = T[idx, :, 1]       # second tangent dir
        n  = N_vec[idx]         # normal dir

        # Tangent directions (two line segments)
        for t_vec in (t1, t2):
            p_start = p - 0.5 * scale_tangent * t_vec
            p_end   = p + 0.5 * scale_tangent * t_vec

            fig.add_trace(go.Scatter3d(
                x=[p_start[0], p_end[0]],
                y=[p_start[1], p_end[1]],
                z=[p_start[2], p_end[2]],
                mode="lines",
                name="Tangent basis" if first_tan else None,
                showlegend=first_tan,
                line=dict(width=4),
            ))
            first_tan = False

        # Normal direction
        p_n_start = p
        p_n_end   = p + scale_normal * n

        fig.add_trace(go.Scatter3d(
            x=[p_n_start[0], p_n_end[0]],
            y=[p_n_start[1], p_n_end[1]],
            z=[p_n_start[2], p_n_end[2]],
            mode="lines",
            name="Normal" if first_norm else None,
            showlegend=first_norm,
            line=dict(width=4, dash="dot"),
        ))
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

### Nicky 
# mainfold_approx.py
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
