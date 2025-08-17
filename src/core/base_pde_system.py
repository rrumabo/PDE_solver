from utils.diagnostic_manager import DiagnosticManager
from src.core.time_integrators import rk4_step

class BasePDESystem:
    def __init__(self, rhs_func, step_func=None, diagnostic_manager=None):
        if step_func is None:
            step_func = rk4_step
        self.rhs_func = rhs_func
        self.step_func = step_func
        self.diagnostic_manager = diagnostic_manager
        self.dimension = None  # Will be auto-detected at runtime

    def evolve(self, u0, dt, steps):
        original_shape = u0.shape
        # Detect and store dimensionality
        if u0.ndim == 1:
            self.dimension = 1
        elif u0.ndim == 2:
            self.dimension = 2
        else:
            raise ValueError(f"Unsupported input dimension: {u0.ndim}")
        print(f"[BasePDESystem] Detected problem dimension: {self.dimension}D")
        u = u0.flatten().copy()  # internally use flattened 1D arrays
        u_history = [u.copy()]

        for step in range(steps):
            t = step * dt
            result = self.step_func(u, t, dt, self.rhs_func, self.diagnostic_manager)
            if isinstance(result, tuple):
                u, residual = result
            else:
                u = result
                residual = None

            if self.diagnostic_manager:
                self.diagnostic_manager.track_step(u, t, residual=residual)

            u_history.append(u.copy())

        # Error handling: check that output shape remains consistent
        for u_t in u_history:
            if u_t.size != u0.size:
                raise ValueError("Mismatch in size during evolution. Check RHS or integrator.")

        # Reshape output history to original user input shape
        return [u_t.reshape(original_shape) for u_t in u_history]

def run_simulation(pde_system, u0, T, dt):
    steps = int(T / dt)
    return pde_system.evolve(u0, dt, steps)

import numpy as np

class NLSEITESystem1D(BasePDESystem):
    def __init__(self, laplacian_op, dt, renormalize=True, diagnostic_manager=None):
        if not (hasattr(laplacian_op, "shape") and hasattr(laplacian_op, "__matmul__")):
            raise TypeError("Laplacian operator must support matrix multiplication and have a shape attribute.")
        if laplacian_op.ndim != 2 or laplacian_op.shape[0] != laplacian_op.shape[1]:
            raise ValueError("Laplacian must be a square 2D matrix.")

        self.renormalize = renormalize

        def rhs_func(psi, t):
            if psi.ndim != 1:
                raise ValueError("Input state ψ must be a 1D array.")
            nonlinear_term = np.abs(psi) ** 2 * psi
            return -(laplacian_op @ psi + nonlinear_term)

        def step_func(u, rhs_func, t, dt):
            u_next = u + dt * rhs_func(u, t)
            if renormalize:
                norm = np.linalg.norm(u_next)
                if norm > 0:
                    u_next = u_next / norm
            return u_next

        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)
        