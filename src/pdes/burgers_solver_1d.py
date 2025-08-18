import numpy as np
from src.core.pde_systems import BasePDESystem

class BurgersSolver1D(BasePDESystem):

    def __init__(self, L_op, dx, nu=0.01, bc: str = "periodic", step_func=None, diagnostic_manager=None):
        self.L_op = L_op               # (N x N) sparse/dense Laplacian
        self.dx = float(dx)
        self.nu = float(nu)
        self.bc = str(bc).lower()
        self.dimension = 1
        self.step_func = step_func

        def rhs_func(u, t):
            # --- Rusanov/LLF flux for F(u)=0.5*u^2 ---
            if self.bc == "periodic":
                up = np.roll(u, -1)              # u_{i+1}
                F  = 0.5 * u * u
                Fp = 0.5 * up * up
                a  = np.maximum(np.abs(u), np.abs(up))
                F_iphalf = 0.5 * (F + Fp) - 0.5 * a * (up - u)
                F_imhalf = np.roll(F_iphalf, 1)
            else:
                # simple one-sided ghosting to keep shape; diffusion BC handled by L_op
                up = np.r_[u[1:], u[-1]]
                F  = 0.5 * u * u
                Fp = 0.5 * up * up
                a  = np.maximum(np.abs(u), np.abs(up))
                F_iphalf = 0.5 * (F + Fp) - 0.5 * a * (up - u)
                F_imhalf = np.r_[F_iphalf[0], F_iphalf[:-1]]

            adv = -(F_iphalf - F_imhalf) / self.dx
            diff = self.nu * (self.L_op @ u)
            return adv + diff

        # delegate to BasePDESystem for time-stepping/diagnostics
        super().__init__(rhs_func=rhs_func, step_func=step_func, diagnostic_manager=diagnostic_manager)

    # Provide a local run wrapper for convenience/compatibility
    def run(self, u0, T, dt):
        # If BasePDESystem already offers run, prefer that
        base_run = getattr(super(), "run", None)
        if callable(base_run):
            return base_run(u0=u0, T=T, dt=dt)
        # Fallback: minimal loop using the provided step_func
        if self.step_func is None:
            raise ValueError("step_func must be provided for time integration")
        u = np.array(u0, dtype=float).copy()
        t = 0.0
        out = [u.copy()]
        times = [t]
        nsteps = int(np.ceil(T / dt))
        step = self.step_func
        for _ in range(nsteps):
            u = step(u, self.rhs_func, t, dt)  # rhs_func is bound in BasePDESystem
            t += dt
            out.append(u.copy())
            times.append(t)
        return out, np.asarray(times)