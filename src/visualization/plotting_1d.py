import matplotlib.pyplot as plt
import os

def plot_initial_final(x, u0, u_final, title="Initial vs Final", save_path=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, u0, label="Initial", linestyle="--")
    ax.plot(x, u_final, label="Final", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if ylim is not None:
        ax.set_ylim(ylim)
    if save_path:
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()
    return fig, ax