"""
Concepto: curvas de nivel (contour plot) con el campo de gradiente
proyectado encima, para visualizar que el gradiente en cada punto es
perpendicular a la curva de nivel que pasa por ese punto y apunta en la
dirección de máximo crecimiento de f.
f(x,y) = x^2 + 2y^2 ;  grad f = (2x, 4y)
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: np.sin(x) * np.cos(y)
grad = lambda x, y: (np.cos(x) * np.cos(y), -np.sin(x) * np.sin(y))

x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# malla mas gruesa para los vectores gradiente
xg = np.linspace(-2.5, 2.5, 9)
yg = np.linspace(-2.5, 2.5, 9)
XG, YG = np.meshgrid(xg, yg)
U, V = grad(XG, YG)

fig, ax = plt.subplots(figsize=(6.5, 6))
cs = ax.contour(X, Y, Z, levels=12, cmap="viridis")
ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")

# normalizamos solo para que las flechas tengan longitud comparable
mag = np.sqrt(U**2 + V**2)
mag[mag == 0] = 1
ax.quiver(XG, YG, U / mag, V / mag, mag, cmap="autumn",
          scale=18, width=0.006)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"Curvas de nivel de $f(x,y)=\sin(x)\cos(y)$ y campo $\nabla f$"
             "\n(el gradiente es perpendicular a cada curva de nivel)")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(savepath("fig03_curvas_nivel_gradiente.png"), dpi=150)
print("OK ->", savepath("fig03_curvas_nivel_gradiente.png"))
    