import matplotlib.pyplot as plt
import os
from matplotlib import animation
import numpy as np

def plot_final_frame(u, x, y, title="Final State", cmap="viridis", save_path=None, vmin=None, vmax=None):
    if u.ndim == 1:
        u = u.reshape((len(y), len(x)))
    fig, ax = plt.subplots()
    im = ax.imshow(u, origin="lower", extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())), cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig, ax

def plot_initial_vs_final(u0, u1, x, y, titles=("Initial", "Final"), cmap="viridis", save_path=None, vmin=None, vmax=None):
    if u0.ndim == 1:
        u0 = u0.reshape((len(y), len(x)))
    if u1.ndim == 1:
        u1 = u1.reshape((len(y), len(x)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    im0 = axes[0].imshow(u0, origin="lower", extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(titles[0])
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u1, origin="lower", extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())), cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(titles[1])
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig, axes

def animate_2d(u_history, x, y, interval=40, filename="figures/heat_2D.mp4", cmap="viridis", vmin=None, vmax=None, stride=1, fps=20, dpi=120, dt=None):
    # Select frames with stride
    u_history = u_history[::stride]

    # Reshape frames if needed
    reshaped_frames = []
    for u in u_history:
        if u.ndim == 1:
            reshaped_frames.append(u.reshape((len(y), len(x))))
        else:
            reshaped_frames.append(u)
    u_history = reshaped_frames

    fig, ax = plt.subplots(dpi=dpi)
    img = ax.imshow(u_history[0], extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    def update(frame):
        img.set_data(u_history[frame])
        return [img]

    anim = animation.FuncAnimation(fig, update, frames=len(u_history), interval=interval, blit=True)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        anim.save(filename, writer="ffmpeg", fps=fps, dpi=dpi)
        plt.close(fig)
    return anim