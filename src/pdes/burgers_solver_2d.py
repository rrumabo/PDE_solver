

import numpy as np
from src.core.pde_systems import BasePDESystem

class BurgersSolver2D(BasePDESystem):
    def __init__(self, L_op, dx, dy, nu=0.01, step_func=None, diagnostic_manager=None):
        """
        Solves the 2D viscous Burgers' equation:
            ∂u/∂t + u·∇u = ν Δu

        Args:
            L_op (ndarray): 2D Laplacian matrix (flattened Nx*Ny x Nx*Ny)
            dx (float): Grid spacing in x-direction
            dy (float): Grid spacing in y-direction
            nu (float): Viscosity coefficient
            step_func: Time integration method
            diagnostic_manager: Optional diagnostic manager
        """
        self.L_op = L_op
        self.nu = nu
        self.dx = dx
        self.dy = dy
        self.dimension = 2
        print("[BurgersSolver2D] Initialized with 2D configuration.")

        def rhs_func(u_flat, t):
            N = int(np.sqrt(u_flat.size))  # assuming Nx = Ny
            u = u_flat.reshape((N, N))

            # Compute gradients using central differences
            dudx = np.zeros_like(u)
            dudy = np.zeros_like(u)

            dudx[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
            dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dy)

            # Periodic BCs
            dudx[0, :] = (u[1, :] - u[-1, :]) / (2 * dx)
            dudx[-1, :] = (u[0, :] - u[-2, :]) / (2 * dx)
            dudy[:, 0] = (u[:, 1] - u[:, -1]) / (2 * dy)
            dudy[:, -1] = (u[:, 0] - u[:, -2]) / (2 * dy)

            convection = u * dudx + u * dudy
            diffusion = self.nu * (self.L_op @ u_flat)

            return -convection.flatten() + diffusion

        super().__init__(rhs_func=rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)