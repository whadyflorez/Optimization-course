"""
Concepto: introducción a restricciones de DESIGUALDAD y condiciones KKT.
    min  f(x,y) = (x-3)^2 + (y-2)^2
    s.a. g(x,y) = x + y - 3 <= 0,   x >= 0,   y >= 0

El mínimo SIN restricciones sería (3,2), que es infactible (3+2=5>3), así
que la restricción g(x,y)<=0 queda ACTIVA en el óptimo. Resolviendo la
condición de estacionariedad de KKT sobre esa restricción activa:
    grad f + mu * grad g = 0 ,  mu >= 0
se obtiene x*=(2,1), mu=2 (mu>=0 confirma que es un mínimo KKT válido).
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: (x - 3) ** 2 + (y - 2) ** 2
grad_f = lambda x, y: np.array([2 * (x - 3), 2 * (y - 2)])
grad_g = np.array([1.0, 1.0])  # g(x,y) = x+y-3

x_unc = (3.0, 2.0)     # óptimo sin restricciones (infactible)
x_opt = (2.0, 1.0)     # óptimo KKT (restricción activa x+y=3)
mu = 2.0

x = np.linspace(-0.5, 4, 300)
y = np.linspace(-0.5, 4, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(6.8, 6))

# región factible: x>=0, y>=0, x+y<=3  (triángulo)
feas_x = [0, 3, 0, 0]
feas_y = [0, 0, 3, 0]
ax.fill(feas_x, feas_y, color="lightgreen", alpha=0.4,
        label="región factible", zorder=0)

cs = ax.contour(X, Y, Z, levels=14, cmap="viridis", zorder=1)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

xr = np.linspace(0, 3, 50)
ax.plot(xr, 3 - xr, color="crimson", lw=2.2,
        label=r"restricción activa $x+y=3$")

ax.plot(*x_unc, "s", color="gray", ms=8, zorder=5)
ax.annotate("óptimo sin\nrestricciones\n(infactible)", x_unc,
            textcoords="offset points", xytext=(10, 5), fontsize=8, color="gray")

ax.plot(*x_opt, "ko", ms=9, zorder=6)
ax.annotate(r"$x^*=(2,1)$" "\n(óptimo KKT)", x_opt,
            textcoords="offset points", xytext=(-70, -35), fontsize=9)

gf = grad_f(*x_opt); gf_n = gf / np.linalg.norm(gf)
gg_n = grad_g / np.linalg.norm(grad_g)
ax.quiver(*x_opt, *gf_n, color="blue", scale=8, width=0.012,
          label=r"$\nabla f(x^*)$")
ax.quiver(*x_opt, *(-gg_n), color="darkorange", scale=8, width=0.007,
          label=r"$-\nabla g(x^*)$  (dirección opuesta: $\nabla f=-\mu\nabla g,\ \mu>0$)")

ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("KKT con restricción de desigualdad activa")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_xlim(-0.5, 4); ax.set_ylim(-0.5, 4)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(savepath("fig11_kkt_desigualdad.png"), dpi=150)
print("OK -> x*=(2,1), mu=2")
