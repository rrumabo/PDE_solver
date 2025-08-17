import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np

def save_animation_1d(x, u_history, dt=0.001, save_path="figures/heat_diffusion.mp4", stride=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    u_frames = u_history[::stride]  # Only use every `stride` frame

    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot(x, u_frames[0], color='blue')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1.1 * max(np.max(u) for u in u_frames))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Heat Diffusion Over Time")

    def update(frame):
        line.set_ydata(u_frames[frame])
        ax.set_title(f"Heat Diffusion at t = {frame * stride * dt:.3f}")
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(u_frames), interval=50, blit=True)

    if save_path:
        ani.save(save_path, writer="ffmpeg", fps=20)
        plt.close()

    return ani, save_path