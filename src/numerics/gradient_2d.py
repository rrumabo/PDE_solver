import numpy as np

def make_gradient_2d(Nx, Ny, dx, dy, *, bc: str = "periodic"):
    """Return a function grad(u_flat)->(dudx_flat[:,None], dudy_flat[:,None]).
    bc: 'periodic', 'dirichlet' or 'neumann' (central diffs; 1-sided at boundaries for dirichlet and neumann)."""
    bc = bc.lower()
    if bc not in ("periodic", "dirichlet", "neumann"):
        raise ValueError(f"Unsupported bc: {bc}")

    def gradient(u_flat: np.ndarray):
        u = u_flat.reshape(Nx, Ny)

        if bc == "periodic":
            dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
            dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
        elif bc == "dirichlet":
            dudx = np.empty_like(u)
            dudy = np.empty_like(u)
            # x-derivative
            dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dx)
            dudx[0,  :]   = (u[1,  :] - u[0,  :]) / dx         # forward
            dudx[-1, :]   = (u[-1, :] - u[-2, :]) / dx         # backward
            # y-derivative
            dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
            dudy[:, 0]    = (u[:, 1]  - u[:, 0])  / dy         # forward
            dudy[:, -1]   = (u[:, -1] - u[:, -2]) / dy         # backward
        else:  # neumann
            dudx = np.empty_like(u)
            dudy = np.empty_like(u)
            # x-derivative
            dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dx)
            dudx[0, :] = 0.0
            dudx[-1, :] = 0.0
            # y-derivative
            dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
            dudy[:, 0] = 0.0
            dudy[:, -1] = 0.0

        return dudx.reshape(-1, 1), dudy.reshape(-1, 1)

    return gradient