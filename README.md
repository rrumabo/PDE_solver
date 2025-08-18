# PDE_solver

A clean, modular sandbox for solving classical PDEs in Python.  
The goal is to demonstrate solid numerical methods, readable code, and a professional project structure suitable for research demos and extensions.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Features (current)

- **Uniform 1D/2D grids** with finite-difference operators
- **Time integration schemes**: Euler, RK4 (others can be added)
- **Boundary conditions**: Dirichlet, periodic
- **Diagnostics**: mass, L² norm, extrema, error metrics
- **Visualization utilities**: static plots + MP4/HTML5 animations
- **Config-driven runs** via `config.yaml` and `main.py`

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a demo
## Running Simulations

This project provides a simple CLI interface for running different PDE solvers.  
All runs will generate outputs (plots, diagnostics, logs) inside `outputs/`.

### Heat Equation (1D)
```bash
python main.py --pde heat --dim 1 --verbose
```

### Heat Equation (2D)
```bash
python main.py --pde heat --dim 2 --verbose
```

### Burgers’ Equation (1D)
```bash
python main.py --pde burgers --dim 1 --verbose
```
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Roadmap

The solver is being expanded systematically across classical PDE families.
Planned milestones:
	•	v2.x: 2D heat equation (done), extended diagnostics (done)
	•	v3.x: Advection–diffusion equation (transport + diffusion)
	•	v4.x: Burgers’ equation (viscous and inviscid shocks)
	•	v5.x: Nonlinear Schrödinger equation (NLSE) in 1D/2D
	•	v5.x+: Imaginary-time evolution (ground states)
	•	v6.x: Shallow-water or Serre equations (geophysical flows)
	•	Later: Integration with μ-stabilized feedback controller for research applications in attractor dynamics and forecasting.
