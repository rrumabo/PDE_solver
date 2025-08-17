

import numpy as np
from src.core.base_pde_system import BasePDESystem

class NLSEITESystem1D(BasePDESystem):
    """
    Imaginary-Time Evolution (ITE) system for the 1D Nonlinear Schrödinger Equation (NLSE):
        ∂ψ/∂τ = Δψ − |ψ|² ψ
    Used to compute ground states via evolution toward minimum energy configurations.
    """

    def __init__(self, laplacian_op, step_func, diagnostic_manager=None):
        if not (hasattr(laplacian_op, "shape") and hasattr(laplacian_op, "__matmul__")):
            raise TypeError("Laplacian must support matrix multiplication and have a shape.")
        if laplacian_op.ndim != 2 or laplacian_op.shape[0] != laplacian_op.shape[1]:
            raise ValueError("Laplacian must be a square matrix.")

        def rhs_func(psi, t):
            if psi.ndim != 1:
                raise ValueError("State ψ must be a 1D array.")
            nonlinear_term = np.abs(psi)**2 * psi
            return laplacian_op @ psi - nonlinear_term

        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)