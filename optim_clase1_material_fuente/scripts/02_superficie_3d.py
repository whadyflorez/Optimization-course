"""
Concepto: grafica de una superficie z = f(x,y).
f(x,y) = x^2 + 2y^2  (paraboloide elíptico, función objetivo de ejemplo
que se reutiliza en varias figuras de la clase: gradiente, curvas de
nivel, convexidad, punto crítico único en el origen).
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: x**2 + 2 * y**2

x = np.linspace(-3, 3, 120)
y = np.linspace(-3, 3, 120)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(7, 5.5))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.9,
                        linewidth=0, antialiased=True, rstride=2, cstride=2)
ax.contour(X, Y, Z, zdir="z", offset=Z.min() - 5, cmap="viridis", levels=10)

ax.plot([0], [0], [f(0, 0)], "o", color="red", ms=8)
ax.text(0, 0, f(0, 0) + 3, "mínimo global\n(0,0)", color="red", fontsize=9)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
ax.set_title(r"Superficie $f(x,y) = x^2 + 2y^2$")
fig.colorbar(surf, shrink=0.55, pad=0.08, label="f(x,y)")
fig.tight_layout()
fig.savefig(savepath("fig02_superficie_3d.png"), dpi=150)
print("OK ->", savepath("fig02_superficie_3d.png"))
