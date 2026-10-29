import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _util import fig_path

plt.rcParams.update({"font.size": 13})

fig, ax = plt.subplots(figsize=(6.0, 5.4))

# x1<=4, x2<=4, x1+x2<=4  -> triangulo (0,0),(4,0),(0,4)
verts = [(0, 0), (4, 0), (0, 4)]
poly = Polygon(verts, closed=True, facecolor="#8C1D40", alpha=0.15,
                edgecolor="#8C1D40", linewidth=2, zorder=1)
ax.add_patch(poly)

x = np.linspace(-0.5, 5.5, 400)
ax.axvline(4, color="#3B7EA1", lw=1.6, ls="--", alpha=0.85, label=r"$x_1=4$")
ax.plot(x, np.full_like(x, 4), color="#2E7D32", lw=1.6, ls="--", alpha=0.85, label=r"$x_2=4$")
ax.plot(x, 4 - x, color="#6A3D9A", lw=1.6, ls="--", alpha=0.85, label=r"$x_1+x_2=4$")

for (vx, vy) in verts:
    ax.plot(vx, vy, "o", color="black", ms=5, zorder=3)

ax.plot(0, 4, "o", color="#D2691E", ms=11, mfc="none", mew=2.5, zorder=5)
ax.annotate("Vértice (0,4): convergen 3 restricciones"
            "\n"
            r"($x_2 \leq 4$, $x_1+x_2 \leq 4$, $x_1 \geq 0$)"
            "\n→ vértice DEGENERADO",
            xy=(0, 4), xytext=(0.35, 4.9), fontsize=9.7, color="#8B4000")

path = [(0, 0), (0, 4)]
ax.annotate("", xy=path[1], xytext=path[0],
            arrowprops=dict(arrowstyle="-|>", color="#D2691E", lw=2.6,
                             shrinkA=8, shrinkB=8), zorder=4)
ax.annotate(r"Empate en la razón: $4/1=4/1$", xy=(0.05, 2), fontsize=9.5,
            rotation=90, va="center", color="#333333")

ax.set_xlim(-1.3, 5.0)
ax.set_ylim(-0.8, 6.3)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Ejemplo degenerado: región y vértice de empate")
ax.legend(loc="center right", fontsize=9, framealpha=0.95, bbox_to_anchor=(1.0, 0.32))
ax.set_aspect("equal")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(fig_path("fig02_region_degenerada.png"), dpi=180)
print("OK fig02")
