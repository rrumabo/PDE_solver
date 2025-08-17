import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union

# 2D Laplacian via Kronecker sums; periodic or dirichlet BC.
# Accepts either: (Nx, Ny, dx, dy) or ((Nx,Ny), (dx,dy)).

def make_laplacian_2d(
    Nx: Union[int, Tuple[int, int]],
    Ny: Union[int, None] = None,
    dx: Union[float, Tuple[float, float], None] = None,
    dy: Union[float, None] = None,
    *,
    bc: str = "periodic",
    fmt: str = "csr",
):
    # Unpack flexible args
    if Ny is None:
        # assume Nx=(Nx,Ny), dx=(dx,dy)
        assert isinstance(Nx, (tuple, list)) and dx is not None and isinstance(dx, (tuple, list)), "use (Nx,Ny),(dx,dy) or Nx,Ny,dx,dy"
        Nx, Ny = int(Nx[0]), int(Nx[1])
        dx, dy = float(dx[0]), float(dx[1])
    else:
        Nx, Ny = int(Nx), int(Ny)  # type: ignore[arg-type]
        assert dx is not None and dy is not None, "dx and dy required"
        # in this branch dx, dy must be scalars; guard against tuple input
        if isinstance(dx, (tuple, list)) or isinstance(dy, (tuple, list)):
            raise TypeError("Use make_laplacian_2d((Nx,Ny), (dx,dy), ...) for tuple inputs")
    dx, dy = float(dx), float(dy)

    bc = bc.lower()
    fmt = fmt.lower()

    Ix = sp.eye(Nx, format=fmt)
    Iy = sp.eye(Ny, format=fmt)

    # 1D Laplacians
    e_x = np.ones(Nx)
    Dx = sp.diags([-2*e_x, e_x, e_x], offsets=(0, -1, 1), shape=(Nx, Nx), format=fmt)
    if bc == "periodic":
        Dx = Dx.tolil(); Dx[0, -1] = 1.0; Dx[-1, 0] = 1.0; Dx = Dx.asformat(fmt)
    elif bc != "dirichlet":
        raise ValueError(f"Unsupported bc: {bc}")
    Dx = Dx * (1.0 / (dx*dx))

    e_y = np.ones(Ny)
    Dy = sp.diags([-2*e_y, e_y, e_y], offsets=(0, -1, 1), shape=(Ny, Ny), format=fmt)
    if bc == "periodic":
        Dy = Dy.tolil(); Dy[0, -1] = 1.0; Dy[-1, 0] = 1.0; Dy = Dy.asformat(fmt)
    Dy = Dy * (1.0 / (dy*dy))

    # Kronecker sum
    L = sp.kron(Iy, Dx, format=fmt) + sp.kron(Dy, Ix, format=fmt)
    return L