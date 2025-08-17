import numpy as np
from scipy.sparse import diags

def make_laplacian_1d(N, dx):
    """
    Construct the 1D Laplacian matrix with periodic boundary conditions.

    Args:
        N (int): Number of grid points.
        dx (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse (N x N) Laplacian matrix.
    """
    diagonals = [
        -2.0 * np.ones(N),
        np.ones(N - 1),
        np.ones(N - 1)
    ]
    offsets = (0, -1, 1)
    L = diags(diagonals, offsets, shape=(N, N), format="csr")

    # Periodic boundaries
    L = L.tolil()
    L[0, -1] = 1.0
    L[-1, 0] = 1.0
    return (1.0 / dx**2) * L.tocsr()

def make_laplacian_1d_dirichlet(N, dx):
    """
    Construct the 1D Laplacian matrix with Dirichlet boundary conditions.

    Args:
        N (int): Number of interior grid points (excluding boundaries).
        dx (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse (N x N) Laplacian matrix with Dirichlet BCs.
    """
    diagonals = [
        -2.0 * np.ones(N),
        np.ones(N - 1),
        np.ones(N - 1)
    ]
    offsets = (0, -1, 1)
    L = diags(diagonals, offsets, shape=(N, N), format="csr")
    return (1.0 / dx**2) * L