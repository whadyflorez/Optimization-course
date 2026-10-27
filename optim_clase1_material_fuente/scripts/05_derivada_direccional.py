"""
Concepto: derivada direccional D_u f(x0) = grad f(x0) . u  (u unitario).
Panel izquierdo: curvas de nivel de f, punto x0 y dirección u.
Panel derecho: corte 1D  g(t) = f(x0 + t*u)  y su recta tangente en t=0,
cuya pendiente es exactamente D_u f(x0).
f(x,y) = x^2 + 2y^2
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: x**2 + 2 * y**2
grad = lambda x, y: np.array([2 * x, 4 * y])

x0 = np.array([1.0, 1.0])
theta = np.deg2rad(60)                       # dirección elegida
u = np.array([np.cos(theta), np.sin(theta)])  # vector unitario
Duf = grad(*x0) @ u                           # derivada direccional

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

# --- panel izquierdo: curvas de nivel + dirección ---
xs = np.linspace(-1, 3, 300)
ys = np.linspace(-1, 3, 300)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)
cs = ax1.contour(X, Y, Z, levels=12, cmap="viridis")
ax1.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
ax1.plot(*x0, "ko", ms=7)
ax1.annotate(r"$x_0=(1,1)$", x0, textcoords="offset points", xytext=(8, 6))
ax1.quiver(*x0, *u, color="crimson", scale=6, width=0.01,
           label=r"dirección $u$")
gvec = grad(*x0)
gnorm = gvec / np.linalg.norm(gvec)
ax1.quiver(*x0, *gnorm, color="blue", scale=6, width=0.01,
           label=r"$\nabla f(x_0)$ (normalizado)")
ax1.set_title("Punto $x_0$, dirección $u$ y gradiente")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend(fontsize=8, loc="lower right")
ax1.set_aspect("equal")

# --- panel derecho: corte 1D g(t) = f(x0 + t u) ---
t = np.linspace(-1.5, 1.5, 300)
g = np.array([f(*(x0 + ti * u)) for ti in t])
tangente = f(*x0) + Duf * t

ax2.plot(t, g, color="#1f77b4", lw=2.2, label=r"$g(t)=f(x_0+tu)$")
ax2.plot(t, tangente, "--", color="crimson", lw=2,
         label=fr"tangente en $t=0$, pendiente $D_uf(x_0)={Duf:.2f}$")
ax2.plot([0], [f(*x0)], "ko", ms=6)
ax2.set_xlabel("t")
ax2.set_ylabel("g(t)")
ax2.set_title("Corte unidimensional a lo largo de u")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(savepath("fig05_derivada_direccional.png"), dpi=150)
print("OK -> Duf =", Duf)
