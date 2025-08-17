import numpy as np
import matplotlib.pyplot as plt

# Re-export  manager 
from .diagnostic_manager import DiagnosticManager

try:
    from scipy.integrate import simpson as _simpson_1d  
except Exception:
    _simpson_1d = None  


def _mass_1d(u, dx):
    """1D mass: Simpson if SciPy is available; otherwise rectangle."""
    u = np.real_if_close(np.asarray(u))
    if _simpson_1d is not None:
        try:
            return float(_simpson_1d(u, dx=dx))
        except Exception:
            pass
    return float(u.sum() * dx)


def _mass_2d(u, dx, dy):
    """2D mass: rectangle rule (robust for uniform/periodic grids)."""
    u = np.real_if_close(np.asarray(u))
    return float(u.sum() * dx * dy)


def plot_mass_evolution(u_history, dx, dy=None, times=None, title="Mass over time"):
    """
    Plot total mass vs time.
    - 1D (dy is None OR frames are 1D): Simpson (fallback to rectangle if SciPy missing)
    - 2D (dy given AND frames are 2D):  rectangle rule
    Returns: (fig, ax, masses)
    """
    frames = [np.asarray(u) for u in u_history]
    masses = []

    if dy is None or (len(frames) > 0 and frames[0].ndim == 1):
        masses = [_mass_1d(u, dx) for u in frames]
    else:
        masses = [_mass_2d(u, dx, float(dy)) for u in frames]

    if times is None:
        times = np.arange(len(frames))

    fig, ax = plt.subplots()
    ax.plot(list(times), masses, label="mass")
    ax.set_xlabel("time")
    ax.set_ylabel("mass")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return fig, ax, masses