

import numpy as np
from src.core.base_pde_system import BasePDESystem

from src.utils.diagnostic_manager import DiagnosticManager

"""
Nonlinear Schrödinger Equation (2D) system:
    i ∂ψ/∂t = -Δψ - |ψ|²ψ
Supports flattened complex input ψ (from 2D grid).
"""

class NLSEPDESystem2D(BasePDESystem):
    def __init__(self, laplacian_op, step_func, diagnostic_manager: DiagnosticManager = None):
        if not (hasattr(laplacian_op, "shape") and hasattr(laplacian_op, "__matmul__")):
            raise TypeError("Laplacian operator must support matrix multiplication and have a shape attribute.")

        if laplacian_op.ndim != 2 or laplacian_op.shape[0] != laplacian_op.shape[1]:
            raise ValueError("Laplacian operator must be a square 2D matrix.")

        def rhs_func(psi_flat, t):
            if psi_flat.ndim != 1:
                raise ValueError("Input state ψ must be a 1D flattened array.")
            nonlinear_term = np.abs(psi_flat) ** 2 * psi_flat
            return -1j * (laplacian_op @ psi_flat + nonlinear_term)

        super().__init__(rhs_func=rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)