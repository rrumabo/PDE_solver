def make_laplacian(dim, N, dx, dy=None, bc="periodic"):
    if dim == 1:
        from .laplacian_1d import make_laplacian_1d
        return make_laplacian_1d(N, dx, bc=bc)
    elif dim == 2:
        from .laplacian_2d import make_laplacian_2d
        return make_laplacian_2d(N, N, dx, dy or dx, bc=bc)
    else:
        raise NotImplementedError("Only 1D and 2D supported for now.")