
import numpy as np

def make_advection_matrix_1d(N, dx, scheme="central"):

    A = np.zeros((N, N))

    if scheme == "central":
        for i in range(N):
            A[i, (i - 1) % N] = -0.5 / dx
            A[i, (i + 1) % N] = 0.5 / dx
    elif scheme == "upwind":
        for i in range(N):
            A[i, i] = -1.0 / dx
            A[i, (i - 1) % N] = 1.0 / dx
    else:
        raise ValueError("Unknown scheme. Use 'central' or 'upwind'.")

    return A