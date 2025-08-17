import numpy as np
from src.utils.diagnostic_manager import DiagnosticManager
from src.core.base_pde_system import BasePDESystem

class ExplicitPDESystem2D(BasePDESystem):
    def __init__(self, rhs_func, step_func=None, diagnostic_manager=None):
        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)

class LinearPDESystem2D(BasePDESystem):
    def __init__(self, L_op, alpha=1.0, step_func=None, diagnostic_manager=None):
        self.L_op = L_op
        self.alpha = alpha

        if step_func is None:
            raise ValueError("A valid step_func (time integrator) must be provided for LinearPDESystem2D.")

        # Type and shape checks for L_op
        if not (hasattr(L_op, "shape") and hasattr(L_op, "__matmul__")):
            raise TypeError("Operator must support matrix multiplication and have a shape attribute (e.g., NumPy array or SciPy sparse matrix).")
        if L_op.ndim != 2 or L_op.shape[0] != L_op.shape[1]:
            # Ensure the operator is square (required for linear PDE evolution)
            raise ValueError("L_op must be a square 2D matrix.")

        def rhs_func(u_flat, t):
            if u_flat.ndim != 1:
                # Flattened state vector required for efficient time stepping
                raise ValueError("Input state u must be a 1D array (flattened).")
            return self.alpha * (self.L_op @ u_flat)

        # Call parent constructor with validated setup
        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)

def make_linear_rhs(operator, alpha=1.0):
    def rhs(u, t):
        return alpha * (operator @ u)
    return rhs

class LinearPDESystem1D(BasePDESystem):
    def __init__(self, L_op, alpha=1.0, step_func=None, diagnostic_manager=None):
        self.L_op = L_op
        self.alpha = alpha

        if step_func is None:
            raise ValueError("A valid step_func (time integrator) must be provided for LinearPDESystem1D.")

        # Type and shape checks for L_op
        if not (hasattr(L_op, "shape") and hasattr(L_op, "__matmul__")):
            raise TypeError("Operator must support matrix multiplication and have a shape attribute (e.g., NumPy array or SciPy sparse matrix).")
        if L_op.ndim != 2 or L_op.shape[0] != L_op.shape[1]:
            # Ensure the operator is square (required for linear PDE evolution)
            raise ValueError("L_op must be a square 2D matrix.")

        def rhs_func(u_flat, t):
            if u_flat.ndim != 1:
                # Flattened state vector required for efficient time stepping
                raise ValueError("Input state u must be a 1D array (flattened).")
            return self.alpha * (self.L_op @ u_flat)

        # Call parent constructor with validated setup
        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)
        
class NLSEPDESystem1D(BasePDESystem):
    def __init__(self, laplacian_op, step_func, diagnostic_manager=None):
        if not (hasattr(laplacian_op, "shape") and hasattr(laplacian_op, "__matmul__")):
            raise TypeError("Operator must support matrix multiplication and have a shape attribute (e.g., NumPy array or SciPy sparse matrix).")
        if laplacian_op.ndim != 2 or laplacian_op.shape[0] != laplacian_op.shape[1]:
            raise ValueError("Laplacian must be a square 2D matrix.")

        def rhs_func(psi, t):
            if psi.ndim != 1:
                raise ValueError("Input state ψ must be a 1D complex array.")
            nonlinear_term = np.abs(psi)**2 * psi
            return -1j * (laplacian_op @ psi + nonlinear_term)

        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)