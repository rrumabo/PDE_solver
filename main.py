import os, sys, argparse, logging
from datetime import datetime
import numpy as np

# make 'src' importable even if CWD isn't repo root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# local imports
from src.utils.config_loader import load_config
from src.utils.diagnostic_manager import DiagnosticManager
from src.visualization.plotting_1d import plot_initial_final as plot_if_1d
from src.visualization.plotting_2d import plot_initial_vs_final as plot_if_2d
from src.pdes.heat_solver_1d import run_heat_solver_1d
from src.pdes.heat_solver_2d import run_heat_solver_2d
from src.pdes.burgers_solver_1d import BurgersSolver1D
from src.numerics.laplacian_1d import make_laplacian_1d, make_laplacian_1d_dirichlet
from src.numerics.laplacian_2d import make_laplacian_2d
from src.initial_conditions.gaussian_1d import gaussian_bump
from src.initial_conditions.gaussian_2d import gaussian_wave_packet
from src.initial_conditions.gaussian_2d import gaussian_bump_2d

from src.core.time_integrators import rk4_step, euler_step, rk4_step_op, euler_step_op

logger = logging.getLogger("pde_main")
logger.setLevel(logging.INFO)

# --- helpers ---
def _mk_outdir(base: str | None = None, tag: str | None = None) -> str:
    root = base or "outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"run_{timestamp}" + (f"_tag_{tag}" if tag else "")
    outdir = os.path.abspath(os.path.join(root, name))
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _pick_step(method: str):
    m = method.lower()
    table = {
        "rk4": rk4_step,
        "euler": euler_step,
        "rk4_op": rk4_step_op,
        "euler_op": euler_step_op,
    }
    if m not in table:
        raise ValueError(f"Unsupported integrator: {method}")
    return table[m]


# --- main entry ---
def main(cfg: dict):
    # config
    pde_type = cfg.get("pde", {}).get("type", "heat").lower()
    dim      = int(cfg.get("dimension", 1))
    bc       = cfg.get("bc", {}).get("type", "periodic").lower()
    integ    = cfg.get("integrator", {}).get("method", "rk4")
    T        = float(cfg.get("time", {}).get("T", 0.1))
    dt       = float(cfg.get("time", {}).get("dt", 1e-4))

    # grid
    g = cfg.get("grid", {})
    N_raw = g.get("N", 128)
    L_raw = g.get("L", 1.0)
    if isinstance(N_raw, (tuple, list)):
        Nx, Ny = int(N_raw[0]), int(N_raw[1])
    else:
        Nx = Ny = int(N_raw)
    if isinstance(L_raw, (tuple, list)):
        Lx, Ly = float(L_raw[0]), float(L_raw[1])
    else:
        Lx = Ly = float(L_raw)
    dx = float(g.get("dx", Lx / Nx))
    dy = float(g.get("dy", Ly / Ny))

    # output
    out_cfg = cfg.get("output", {})
    outdir = _mk_outdir(out_cfg.get("folder"), out_cfg.get("tag"))
    handler = logging.FileHandler(os.path.join(outdir, "log.txt"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    if out_cfg.get("verbose", False):
        logger.addHandler(logging.StreamHandler())

    # pick integrator
    step_func = _pick_step(integ)

    # initial conditions
    ic = cfg.get("initial_condition", {})

    if pde_type == "heat" and dim == 1:
        x = np.linspace(0.0, Lx, Nx, endpoint=False)
        L_op = make_laplacian_1d(Nx, dx) if bc == "periodic" else make_laplacian_1d_dirichlet(Nx, dx)
        amp = float(ic.get("amp", ic.get("amplitude", 1.0)))
        c   = float(ic.get("center", 0.5)) * Lx
        sig = float(ic.get("sigma", 0.1)) * Lx
        u0  = gaussian_bump(x, center=c, sigma=sig, amp=amp)
        # --- RK4 stability clamp for 1D heat (periodic FD Laplacian: lam_max ≈ 4/dx^2) ---
        lam_max = 4.0 / (dx * dx)
        dt_rk4_max = 2.785 / lam_max
        dt_use = min(dt, 0.8 * dt_rk4_max)
        logger.info(f"[heat1d] dt user={dt:.3e}, rk4_max={dt_rk4_max:.3e} → using {dt_use:.3e}")
        res = run_heat_solver_1d(L_op=L_op, u0=u0, T=T, dt=dt_use, step_func=step_func)
        if isinstance(res, tuple) and len(res) == 2:
            u_hist, t_hist = res
        else:
            u_hist = res
            t_hist = np.arange(len(u_hist)) * dt_use
        dm = DiagnosticManager(dx=dx, track=("mass","l2_norm","min","max","mean"))
        for u, t in zip(u_hist, t_hist): dm.track_step(u, t)
        # plot
        plot_if_1d(x, u_hist[0], u_hist[-1], title="Heat 1D — initial vs final", save_path=os.path.join(outdir, "heat1d_initial_final.png"))
        dm.save_csv(os.path.join(outdir, "heat1d_diagnostics.csv"))
        dm.save_yaml(os.path.join(outdir, "heat1d_summary.yaml"))
        logger.info("Done heat 1D → %s", outdir)
        return u_hist
    
    if pde_type == "heat" and dim == 2:
        x = np.linspace(0.0, Lx, Nx, endpoint=False)
        y = np.linspace(0.0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="xy")

        # Operator
        L2 = make_laplacian_2d((Nx, Ny), dx=(dx, dy), bc=bc)
        alpha = float(cfg.get("physics", {}).get("params", {}).get("alpha", 1.0))
        L_op = alpha * L2

        # IC (isotropic Gaussian)
        ic = cfg.get("initial_condition", {})
        amp = float(ic.get("amp", ic.get("amplitude", 1.0)))
        center = ic.get("center", 0.5)
        if isinstance(center, (tuple, list)):
            cx, cy = float(center[0]) * Lx, float(center[1]) * Ly
        else:
            cx = cy = float(center) * Lx  # if scalar, use same frac in x,y
        sigma_cfg = float(ic.get("sigma", 0.1))
        sig = sigma_cfg * min(Lx, Ly)

        u0_field = gaussian_bump_2d(X, Y, x0=cx, y0=cy, sigma=sig, amp=amp)
        u0 = u0_field.reshape(Nx * Ny)

        # RK4 stability clamp for 2D diffusion: lam_max ≈ 4/dx^2 + 4/dy^2
        lam_max = 4.0 / (dx * dx) + 4.0 / (dy * dy)
        dt_rk4_max = 2.785 / (alpha * lam_max)
        dt_use = min(dt, 0.8 * dt_rk4_max)
        logger.info(f"[heat2d] dt user={dt:.3e}, rk4_max={dt_rk4_max:.3e} → using {dt_use:.3e}")

        # Run
        u_hist = run_heat_solver_2d(L_op=L_op, u0=u0, Nx=Nx, Ny=Ny, T=T, dt=dt_use, dx=dx, dy=dy, step_func=step_func)
        t_hist = np.arange(len(u_hist)) * dt_use

        # Diagnostics + plot
        dm = DiagnosticManager(dx=dx, dy=dy, track=("mass","l2_norm","min","max","mean"))
        for u, t in zip(u_hist, t_hist):
            dm.track_step(u.reshape(Ny, Nx), t)

        plot_if_2d(u0_field, u_hist[-1].reshape(Ny, Nx), x, y, save_path=os.path.join(outdir, "heat2d_initial_final.png"))
        dm.save_csv(os.path.join(outdir, "heat2d_diagnostics.csv"))
        dm.save_yaml(os.path.join(outdir, "heat2d_summary.yaml"))
        logger.info("Done heat 2D → %s", outdir)
        return u_hist

    if pde_type == "burgers" and dim == 1:
        x = np.linspace(0.0, Lx, Nx, endpoint=False)

        # Laplacian for viscosity (BC-aware)
        Lb = make_laplacian_1d(Nx, dx) if bc == "periodic" else make_laplacian_1d_dirichlet(Nx, dx)

        # Viscosity + initial condition
        nu = float(cfg.get("physics", {}).get("params", {}).get("nu", 0.01))
        ic_type = ic.get("type", "sine").lower()
        if ic_type == "sine":
            k   = int(ic.get("k", 1))
            amp = float(ic.get("amp", ic.get("amplitude", 1.0)))
            u0  = amp * np.sin(2*np.pi*k*x/Lx)
        elif ic_type == "gaussian":
            amp = float(ic.get("amp", ic.get("amplitude", 1.0)))
            c   = float(ic.get("center", 0.5)) * Lx
            sig = float(ic.get("sigma", ic.get("width", 0.1))) * Lx
            u0  = amp * np.exp(-0.5 * ((x - c)/max(sig, 1e-12))**2)
        else:
            u0 = np.zeros_like(x)

        # Pick an RHS-style integrator for Burgers (operator-aware *_op won't work here)
        if "rk4" in integ.lower():
            step_rhs = rk4_step
        else:
            step_rhs = euler_step
        # Class API
        solver = BurgersSolver1D(L_op=Lb, dx=dx, nu=nu, bc=bc, step_func=step_rhs)
        u_res = solver.run(u0=u0, T=T, dt=dt)
        if not (isinstance(u_res, tuple) and len(u_res) == 2):
            raise TypeError(
                f"BurgersSolver1D.run must return (u_hist, t_hist); got {type(u_res).__name__}"
            )
        u_hist, t_hist = u_res

        # Diagnostics + plot
        dm = DiagnosticManager(dx=dx, track=("mass","l2_norm","min","max","mean"))
        for u, t in zip(u_hist, t_hist):
            dm.track_step(u, t)
        plot_if_1d(
            x, u_hist[0], u_hist[-1],
            title="Burgers 1D — initial vs final",
            save_path=os.path.join(outdir, "burgers_1d_initial_final.png")
        )
        dm.save_csv(os.path.join(outdir, "burgers_1d_diagnostics.csv"))
        dm.save_yaml(os.path.join(outdir, "burgers_1d_summary.yaml"))
        logger.info("Done Burgers 1D → %s", outdir)
        return u_hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimal CLI for PDE_solver (notebooks are primary)")
    ap.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    ap.add_argument("--pde", type=str, help="Override pde.type (heat|burgers)")
    ap.add_argument("--dim", type=int, help="Override dimension (1 or 2)")
    ap.add_argument("--tag", type=str, default="", help="Optional tag for output folder name")
    ap.add_argument("--verbose", action="store_true", help="Log to console")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.pde: cfg.setdefault("pde", {})["type"] = args.pde
    if args.dim: cfg["dimension"] = args.dim
    cfg.setdefault("output", {})["tag"] = args.tag
    cfg["output"]["verbose"] = bool(args.verbose)

    main(cfg)