"""Graficación estándar: curvas de nivel + trayectoria de iterados."""
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(f, traj, xlim, ylim, title, fname, levels=30, log_levels=False,
                     extra_label=None, minimum=None, n_grid=300):
    traj = np.asarray(traj)
    x = np.linspace(*xlim, n_grid)
    y = np.linspace(*ylim, n_grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if log_levels:
        lv = np.geomspace(max(Z.min(), 1e-6), Z.max(), levels)
        cs = ax.contour(X, Y, Z, levels=lv, cmap="viridis", linewidths=0.7)
    else:
        cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.7)
    ax.plot(traj[:, 0], traj[:, 1], "o-", color="crimson", markersize=3,
            linewidth=1.2, label=extra_label or "Trayectoria")
    ax.plot(traj[0, 0], traj[0, 1], "s", color="black", markersize=7, label="Inicio")
    if minimum is not None:
        ax.plot(minimum[0], minimum[1], "*", color="gold", markeredgecolor="black",
                markersize=16, label="Mínimo", zorder=5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_convergence(fvals_list, labels, title, fname, ylog=True):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for fv, lab in zip(fvals_list, labels):
        ax.plot(range(len(fv)), fv, marker=".", label=lab)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("$f(x_k)$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
