import numpy as np
from src.utils.diagnostic_manager import DiagnosticManager
from src.core.base_pde_system import BasePDESystem
from src.core.base_pde_system import run_simulation

class AdvectionDiffusionSystem1D(BasePDESystem):
    def __init__(self, advection_op, diffusion_op, step_func=None, diagnostic_manager=None):
        self.advection_op = advection_op
        self.diffusion_op = diffusion_op

        def rhs_func(u, t):
            return - (advection_op @ u) + (diffusion_op @ u)

        super().__init__(rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)
