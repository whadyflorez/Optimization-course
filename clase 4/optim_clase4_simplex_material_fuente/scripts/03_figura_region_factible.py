import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from _util import fig_path

plt.rcParams.update({"font.size": 13})

fig, ax = plt.subplots(figsize=(6.4, 5.6))

# Region factible: x1<=4, 2x2<=12 (x2<=6), 3x1+2x2<=18, x1,x2>=0
verts = [(0, 0), (4, 0), (4, 3), (2, 6), (0, 6)]
poly = Polygon(verts, closed=True, facecolor="#3B7EA1", alpha=0.18,
                edgecolor="#3B7EA1", linewidth=2, zorder=1)
ax.add_patch(poly)

x = np.linspace(-0.5, 7, 400)
ax.axvline(4, color="#8C1D40", lw=1.6, ls="--", alpha=0.8, label=r"$x_1=4$")
ax.plot(x, np.full_like(x, 6), color="#2E7D32", lw=1.6, ls="--", alpha=0.8, label=r"$2x_2=12$")
y3 = (18 - 3 * x) / 2
ax.plot(x, y3, color="#6A3D9A", lw=1.6, ls="--", alpha=0.8, label=r"$3x_1+2x_2=18$")

for (vx, vy) in verts:
    ax.plot(vx, vy, "o", color="black", ms=5, zorder=3)

# Trayectoria del simplex: (0,0) -> (0,6) -> (2,6)
path = [(0, 0), (0, 6), (2, 6)]
for i in range(len(path) - 1):
    x0, y0 = path[i]
    x1_, y1_ = path[i + 1]
    ax.annotate("", xy=(x1_, y1_), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#D2691E", lw=2.6,
                                 shrinkA=8, shrinkB=8), zorder=4)

labels = {(0, 0): "Inicio\n$(x_1,x_2)=(0,0)$\n$z=0$",
          (0, 6): "Iter. 1\n$(0,6)$, $z=30$",
          (2, 6): "Iter. 2 = Óptimo\n$(2,6)$, $z=36$"}
offsets = {(0, 0): (0.35, -0.85), (0, 6): (-1.55, 0.25), (2, 6): (0.15, -1.05)}
for pt, txt in labels.items():
    dx, dy = offsets[pt]
    ax.annotate(txt, xy=pt, xytext=(pt[0] + dx, pt[1] + dy), fontsize=10.5,
                ha="left" if dx > 0 else "left")

ax.set_xlim(-1.8, 6.5)
ax.set_ylim(-1, 7.2)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Región factible y trayectoria del Simplex")
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_aspect("equal")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(fig_path("fig01_region_factible.png"), dpi=180)
print("OK fig01")
