import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import Optional, Tuple
import os

def animate_2d(
    frames,
    x,
    y,
    title: str = "",
    interval: int = 50,
    cmap: str = "viridis",
    extent: Optional[Tuple[float, float, float, float]] = None,
    filename: Optional[str] = None,
    fps: int = 20,
    dpi: int = 120,
    stride: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
):
    """
    Animate a sequence of 2D scalar fields over time using imshow.

    Args:
        frames (list of 2D arrays): The 2D snapshots to animate.
        x (1D array): x-coordinates.
        y (1D array): y-coordinates.
        title (str): Title of the animation.
        interval (int): Delay between frames in milliseconds.
        cmap (str): Matplotlib colormap to use.
        extent (tuple, optional): (xmin, xmax, ymin, ymax) in float. If None, computed from x and y.
        filename (str, optional): Path to save animation. If None, not saved.
        fps (int): Frames per second when saving.
        dpi (int): Dots per inch when saving.
        stride (int): Use every `stride`th frame.
        vmin, vmax (float, optional): Color scale limits.

    Returns:
        anim (FuncAnimation): Matplotlib animation object.
    """
    frames = np.asarray(frames)[::stride]
    if extent is None:
        extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    if vmin is None:
        vmin = float(np.min(frames))
    if vmax is None:
        vmax = float(np.max(frames))

    fig, ax = plt.subplots()
    img = ax.imshow(frames[0], extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(img, ax=ax)

    def update(i):
        img.set_data(frames[i])
        ax.set_title(f"{title} — frame {i}")
        return [img]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval, blit=True
    )

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        try:
            anim.save(filename, fps=fps, dpi=dpi, writer="ffmpeg")
        except Exception:
            gif_name = os.path.splitext(filename)[0] + ".gif"
            anim.save(gif_name, fps=fps, dpi=dpi, writer="pillow")
            print(f"Saved animation to {gif_name}")
        else:
            print(f"Saved animation to {filename}")

    return anim