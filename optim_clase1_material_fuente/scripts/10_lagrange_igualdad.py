"""
Concepto: optimización con restricción de IGUALDAD y multiplicadores de
Lagrange.
    min  f(x,y) = x^2 + y^2         (distancia al origen, al cuadrado)
    s.a. h(x,y) = x + y - 4 = 0     (recta)

Condición necesaria: grad f(x*,y*) = lambda * grad h(x*,y*)
  grad f = (2x, 2y);  grad h = (1,1)  ->  2x = 2y = lambda  ->  x=y
  con x+y=4  ->  x=y=2,  lambda=4
En el óptimo, el gradiente de f es paralelo al gradiente de la restricción
(equivalentemente, perpendicular a la curva de restricción).
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: np.array([2 * x, 2 * y])
grad_h = np.array([1.0, 1.0])  # constante para h(x,y)=x+y-4

x_opt, y_opt = 2.0, 2.0
lam = 4.0

x = np.linspace(-1, 5, 300)
y = np.linspace(-1, 5, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(6.5, 6))
cs = ax.contour(X, Y, Z, levels=14, cmap="viridis")
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

# restricción x + y = 4
xr = np.linspace(-1, 5, 50)
ax.plot(xr, 4 - xr, color="crimson", lw=2.5, label=r"restricción $x+y=4$")

ax.plot([x_opt], [y_opt], "ko", ms=8, zorder=5)
ax.annotate(r"$x^*=(2,2)$", (x_opt, y_opt), textcoords="offset points",
            xytext=(10, 8))

gf = grad_f(x_opt, y_opt)
gf_n = gf / np.linalg.norm(gf)
gh_n = grad_h / np.linalg.norm(grad_h)
ax.quiver(x_opt, y_opt, *gf_n, color="blue", scale=8, width=0.012,
          label=r"$\nabla f(x^*)$")
ax.quiver(x_opt, y_opt, *gh_n, color="darkorange", scale=8, width=0.007,
          label=r"$\nabla h(x^*)$ (misma dirección)")

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(r"Lagrange: en $x^*$, $\nabla f = \lambda \nabla h$"
             fr" ($\lambda={lam:.0f}$)")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(savepath("fig10_lagrange_igualdad.png"), dpi=150)
print("OK -> x*=(2,2), lambda=4")
