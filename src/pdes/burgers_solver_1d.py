

import numpy as np
from src.core.pde_systems import BasePDESystem

class BurgersSolver1D(BasePDESystem):
    def __init__(self, L_op, dx, nu=0.01, step_func=None, diagnostic_manager=None):
        """
        Solves the 1D viscous Burgers' equation:
            du/dt + u du/dx = nu * d^2u/dx^2

        Args:
            L_op (ndarray): Laplacian matrix (Nx x Nx)
            dx (float): Spatial resolution
            nu (float): Viscosity coefficient
            step_func: Time integration method (e.g., rk4_step)
            diagnostic_manager: Optional DiagnosticManager
        """
        self.L_op = L_op
        self.dx = dx
        self.nu = nu
        self.dimension = 1
        print("[BurgersSolver1D] Initialized with 1D configuration.")

        def rhs_func(u, t):
            # Compute du/dx using central differences
            dudx = np.zeros_like(u)
            dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
            # Periodic BCs
            dudx[0] = (u[1] - u[-1]) / (2 * dx)
            dudx[-1] = (u[0] - u[-2]) / (2 * dx)

            convection = u * dudx
            diffusion = nu * (L_op @ u)

            return -convection + diffusion

        super().__init__(rhs_func=rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)