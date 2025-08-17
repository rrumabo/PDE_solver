import os, yaml

# Load + normalize config (supports old/new schemas)

def load_config(path: str = "config.yaml", tag: str | None = None) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg: dict = {}

    grid = raw.get("grid", {})
    dim = grid.get("dim", raw.get("pde", {}).get("dimension", 1))
    L   = grid.get("L", 1.0)
    N   = grid.get("N", 129)
    cfg["grid"] = {"dim": int(dim), "L": float(L), "N": int(N)}

    bc = raw.get("bc", {})
    cfg["bc"] = {"type": str(bc.get("type", "periodic")).lower()}

    time = raw.get("time", {})
    cfg["time"] = {"dt": float(time.get("dt", 1e-4)), "T": float(time.get("T", 1e-1))}

    physics = raw.get("physics", {})
    pde_name = physics.get("pde", raw.get("pde", {}).get("type", "heat")).lower()
    params   = physics.get("params", {})
    if "alpha" in raw and "alpha" not in params:
        params["alpha"] = raw["alpha"]
    cfg["physics"] = {"pde": pde_name, "params": params}

    integ = raw.get("integrator", raw.get("integrator_settings", {}))
    method = integ.get("method", integ.get("name", "rk4")).lower()
    cfg["integrator"] = {"method": method}

    ic = raw.get("initial_condition", raw.get("ic", {}))
    ic_type = ic.get("type", "gaussian").lower()
    center  = ic.get("center", 0.35)
    sigma   = ic.get("sigma", ic.get("width", 0.06))
    amp     = ic.get("amplitude", ic.get("amp", 1.0))
    cfg["initial_condition"] = {"type": ic_type, "center": float(center), "sigma": float(sigma), "amplitude": float(amp)}

    diags = raw.get("diagnostics", [])
    if isinstance(diags, dict):
        diags = [k for k, v in diags.items() if v]
    cfg["diagnostics"] = [str(d).lower() for d in diags]

    io = raw.get("io", raw.get("output", {}))
    cfg["io"] = {
        "outdir": io.get("outdir", io.get("folder", "figures")),
        "save_every": int(io.get("save_every", 1)),
        "tag": (tag if tag is not None else io.get("tag", "run")),
        "verbosity": io.get("verbosity", io.get("level", "normal")),
        "plot_profile": bool(io.get("plot_profile", True)),
        "save_animation": bool(io.get("save_animation", False)),
        "save_diagnostics_csv": bool(io.get("save_diagnostics_csv", True)),
        "save_diagnostics_yaml": bool(io.get("save_diagnostics_yaml", True)),
    }

    cfg["output"] = cfg["io"]  # alias
    cfg["pde"] = {"type": pde_name, "dimension": int(dim)}  # alias
    return cfg

# Minimal validation

def validate_config(cfg: dict) -> None:
    req_top = ["grid", "bc", "time", "physics", "integrator", "initial_condition", "io"]
    for k in req_top:
        if k not in cfg: raise ValueError(f"Missing section: {k}")
    if cfg["grid"]["dim"] not in (1, 2):
        raise ValueError("grid.dim must be 1 or 2")
    if cfg["integrator"]["method"] not in {"rk4", "euler"}:
        raise ValueError(f"Unsupported integrator: {cfg['integrator']['method']}")
    if cfg["physics"]["pde"] not in {"heat", "nlse", "burgers", "advection", "advection_diffusion", "shallow_water"}:
        raise ValueError(f"Unsupported PDE: {cfg['physics']['pde']}")
    if "N" not in cfg["grid"] or "L" not in cfg["grid"]:
        raise ValueError("grid must include N and L")
    ic = cfg["initial_condition"]
    if not {"center", "sigma"}.issubset(ic.keys()):
        raise ValueError("initial_condition requires 'center' and 'sigma'")