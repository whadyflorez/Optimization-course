"""Shared plotting style (thin marks, quiet grid, colour-blind-safe categorical order)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
os.makedirs(FIGS, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8a85", "#222222"
CAT = [BLUE, ORANGE, AQUA, VIOLET, RED]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
    "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.7, "lines.linewidth": 1.8,
    "figure.dpi": 100, "savefig.dpi": 170, "savefig.bbox": "tight", "axes.axisbelow": True,
    "mathtext.fontset": "cm",
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("saved", name)


def grid2(f, lo, hi, n=200):
    """Evaluate a vector function f(x) on a 2-D grid -> X, Y, Z."""
    xs, ys = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[f(np.array([a, b])) for a in xs] for b in ys])
    return X, Y, Z
