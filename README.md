# PDE_solver

A clean, modular sandbox for solving classical PDEs in Python. The goal is to demonstrate solid numerical methods, readable code, and professional project structure suitable for research demos.

## Features (current)
- Uniform 1D/2D grids and finite-difference operators
- Time integration (e.g., RK4 / Euler etc)
- Boundary conditions and source terms (as available)
- Diagnostics: norms / mass conservation / error metrics
- Lightweight plotting utilities

## Quickstart

`bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a demo
python -m src.main --demo heat1d

## Repository Layout

-----------------------------------------------------------------------------------------------------------------------
PDE_solver/
  └─ config.yaml
  └─ main.py
  └─ src/
    └─ .DS_Store
    └─ core/
      └─ base_pde_system.py
      └─ pde_systems.py
      └─ rhea_stabilizer.py
      └─ time_integrators.py
    └─ initial_conditions/
      └─ gaussian_1d.py
      └─ gaussian_2d.py
      └─ profiles_1d.py
    └─ numerics/
      └─ advection_1d.py
      └─ gradient_2d.py
      └─ laplacian_1d.py
      └─ laplacian_2d.py
      └─ laplacian_factory.py
    └─ pdes/
      └─ advection_diffusion_solver_1d.py
      └─ burgers_solver_1d.py
      └─ burgers_solver_2d.py
      └─ heat_solver_1d.py
      └─ heat_solver_2d.py
      └─ nlse_ite_solver_1d.py
      └─ nlse_rhea_1d.py
      └─ nlse_solver_1d.py_
      └─ nlse_solver_2d.py
    └─ utils/
      └─ config_loader.py
      └─ diagnostic_manager.py
      └─ diagnostics.py
    └─ visualization/
      └─ animation_1d.py
      └─ animation_2d.py
      └─ plotting_1d.py
      └─ plotting_2d.py

## License

MIT (or your preferred license).
