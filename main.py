import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys 
import argparse
import csv


from src.utils.diagnostic_manager import DiagnosticManager

from src.utils.config_loader import load_config

from src.visualization.animation_1d import animate_heat_solution

from datetime import datetime

import logging

logger = logging.getLogger("sim_logger")
logger.setLevel(logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# --- Config validation ---
def validate_config(cfg):
    required_keys = ["grid", "pde", "initial_condition", "time", "output", "integrator"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(f"Missing required config section: '{key}'")

    if "type" not in cfg["pde"]:
        raise KeyError("Missing PDE type under 'pde' section")

    valid_pde_types = ["heat", "nlse", "nlse_rhea", "burgers"]
    if cfg["pde"]["type"] not in valid_pde_types:
        raise ValueError(f"Unsupported PDE type: {cfg['pde']['type']}. Supported types: {valid_pde_types}")

    if "L" not in cfg["grid"] or "N" not in cfg["grid"]:
        raise KeyError("Missing 'L' or 'N' in grid configuration")

    if "dt" not in cfg["time"] or "steps" not in cfg["time"]:
        raise KeyError("Missing 'dt' or 'steps' in time configuration")

    if "method" not in cfg["integrator"]:
        raise KeyError("Missing 'method' in integrator configuration")

    valid_methods = ["rk4", "euler"]
    if cfg["integrator"]["method"] not in valid_methods:
        raise ValueError(f"Unsupported integrator method: {cfg['integrator']['method']}. Supported methods: {valid_methods}")

def run_simulation(cfg):
    return main(cfg)

def maybe_plot_final(x, u0, u_final, folder):
    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, label="Initial u₀", linestyle="--")
    plt.plot(x, u_final, label="Final u", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Initial vs Final Heat Profile")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "final_comparison.png"), dpi=300)
    plt.close()

def main(cfg):
    validate_config(cfg)
    dim = cfg.get("dimension", 1)
    init = cfg["initial_condition"]
    out_cfg = cfg["output"]

    pde_cfg = cfg["pde"]
    integrator_cfg = cfg["integrator"]

    L = cfg["grid"]["L"]
    N = cfg["grid"]["N"]
    dx = L / N
    dy = dx  # Assume square grid by default

    from src.core.rhs_examples import make_linear_rhs
    from src.core.time_integrators import rk4_step, euler_step
    if dim == 1:
        import numpy as _np
        from src.numerics.laplacian_1d import make_laplacian_1d
        from src.initial_conditions.profiles_1d import gaussian_bump as ic_func
        from src.core.pde_systems import LinearPDESystem1D

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)
        
        ic_params = {k: v for k, v in init.items() if k != "type"}
        u0 = ic_func(x, **ic_params)
        u0 = u0.reshape(-1)

        from src.core.time_integrators import rk4_step
        pde_system = LinearPDESystem1D(lap, alpha=pde_cfg["alpha"], step_func=rk4_step)

    elif dim == 2:
        from src.numerics.laplacian_2d import make_laplacian_2d
        from src.initial_conditions.gaussian_2d import gaussian_bump_2d as ic_func
        from src.core.pde_systems import LinearPDESystem2D

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        y = x.copy()
        X, Y = np.meshgrid(x, y, indexing="ij")
        lap = make_laplacian_2d(N, N, dx, dx)
        u0 = ic_func(X, Y, center=(init["center"], init["center"]), width=init["width"], amplitude=init["amplitude"])
        u0 = u0.reshape(-1)  

        from src.core.time_integrators import rk4_step
        pde_system = LinearPDESystem2D(lap, alpha=pde_cfg["alpha"], step_func=rk4_step)

    elif pde_cfg["type"] == "nlse":
        from src.numerics.laplacian_1d import make_laplacian_1d
        from src.core.pde_systems import NLSEPDESystem1D 

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)

        ic_params = {k: v for k, v in init.items() if k != "type"}
        psi0 = np.exp(-((x - ic_params.get("center", 0)) / ic_params.get("width", 0.2))**2) * np.exp(1j * ic_params.get("phase", 0.0))
        psi0 = psi0.astype(np.complex128).reshape(-1)

        u0 = psi0  # unify naming convention

        pde_system = NLSEPDESystem1D(lap, alpha=pde_cfg.get("alpha", 1.0), beta=pde_cfg.get("beta", 1.0), step_func=None)

    elif pde_cfg["type"] == "nlse_rhea":
        from src.pdes.nlse_rhea_1d import NLSE_RheaPDESystem1D
        from src.numerics.laplacian_1d import make_laplacian_1d

        x = np.linspace(-L / 2, L / 2, N, endpoint=False)
        lap = make_laplacian_1d(N, dx)

        ic_params = {k: v for k, v in init.items() if k != "type"}
        psi0 = np.exp(-((x - ic_params.get("center", 0)) / ic_params.get("width", 0.2))**2) * np.exp(1j * ic_params.get("phase", 0.0))
        psi0 = psi0.astype(np.complex128).reshape(-1)

        u0 = psi0

        pde_system = NLSE_RheaPDESystem1D(lap, alpha=pde_cfg.get("alpha", 1.0), beta=pde_cfg.get("beta", 1.0), step_func=None, rhea_cfg=cfg.get("rhea", {}))

    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    if integrator_cfg["method"] == "rk4":
        step_func = rk4_step
    elif integrator_cfg["method"] == "euler":
        step_func = euler_step
    else:
        raise ValueError(f"Unsupported integrator method: {integrator_cfg['method']}")

    dt = cfg["time"]["dt"]
    steps = cfg["time"]["steps"]

    u = u0.copy()
    u_history = [u.copy()]
    diagnostics = []

    # Construct dynamic output folder name with timestamp and optional tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = out_cfg.get("tag", "")
    folder_name = f"run_{timestamp}" + (f"_tag_{tag}" if tag else "")
    output_folder = os.path.abspath(os.path.join("outputs", folder_name))
    out_cfg["folder"] = output_folder  # Update config to use this path

    diagnostics_manager = DiagnosticManager(dx=dx, dy=dy if dim == 2 else None, u_ref=u0)

    for step in range(steps):
        rhs_func = pde_system.rhs_func
        t = step * dt
        u = step_func(u, rhs_func, t, dt)
        u_history.append(u.copy())
        diagnostics.append({
            "step": step,
            "min": u.min(),
            "max": u.max(),
            "mean": u.mean(),
        })
        diagnostics_manager.track_step(u, step, rhea=None)

    os.makedirs(output_folder, exist_ok=True)

    log_path = os.path.join(output_folder, "log.txt")
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    if out_cfg.get("verbose", False):
        logger.addHandler(logging.StreamHandler())
    elif out_cfg.get("quiet", False):
        logger.setLevel(logging.ERROR)

    # output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), out_cfg["folder"]))

    if out_cfg.get("save_diagnostics", True):
        log_file = os.path.join(output_folder, "diagnostics.log")
        diagnostics_manager.enable_logging(log_file)

    if out_cfg.get("plot_profile", True):
        maybe_plot_final(x if dim == 1 else X[:,0],
                         u0 if dim == 1 else u0[:,u0.shape[1]//2],
                         u_history[-1],
                         output_folder)

    if out_cfg.get("save_animation", True):
        animate_heat_solution(x if dim == 1 else (x, y),
                              u_history,
                              dt=dt,
                              save_path=os.path.join(output_folder, "heat_diffusion.gif"))

    if out_cfg.get("save_diagnostics", True):
        diagnostics_manager.save_csv(os.path.join(output_folder, "diagnostics_tracked.csv"))
        diagnostics_manager.save_yaml(os.path.join(output_folder, "diagnostics_summary.yaml"))

    logger.info("Simulation complete. Output saved to %s", output_folder)

    return u_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heat solver with configuration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--no-animation", action="store_true", help="Disable saving animation")
    parser.add_argument("--no-diagnostics", action="store_true", help="Disable saving diagnostics")
    parser.add_argument("--no-profile", action="store_true", help="Disable final profile plot")
    parser.add_argument("--pde", type=str, help="Override PDE type (e.g. heat, nlse, burgers)")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for output folder")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output to console")
    parser.add_argument("--quiet", action="store_true", help="Suppress most console output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.pde:
        cfg["pde"]["type"] = args.pde
    if args.no_animation:
        cfg["output"]["save_animation"] = False
    if args.no_diagnostics:
        cfg["output"]["save_diagnostics"] = False
    if args.no_profile:
        cfg["output"]["plot_profile"] = False
    if args.tag:
        cfg.setdefault("output", {})["tag"] = args.tag
    cfg.setdefault("output", {})["verbose"] = args.verbose
    cfg["output"]["quiet"] = args.quiet

    main(cfg)