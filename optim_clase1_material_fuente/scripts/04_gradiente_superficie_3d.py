"""
Concepto: vectores gradiente superpuestos sobre la superficie 3D de f.
En cada punto (x0,y0,f(x0,y0)) se dibuja el vector (∂f/∂x, ∂f/∂y, 0)
"apoyado" sobre la superficie: su proyección horizontal es la dirección
de máximo ascenso, y se ve cómo esos vectores siempre apuntan "cuesta
arriba" hacia el mínimo... en este caso hacia afuera del mínimo (0,0)
porque f crece al alejarse del origen.
f(x,y) = x^2 + 2y^2
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x, y: x**2 + 2 * y**2
grad = lambda x, y: (2 * x, 4 * y)

x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.65, linewidth=0,
                 antialiased=True, rstride=3, cstride=3, zorder=1)

# puntos donde se dibuja el vector gradiente, apoyados sobre la superficie.
# se usa direccion unitaria * longitud fija (no proporcional a |grad f|)
# para que ningun vector se salga del dominio graficado.
pts = [(-2, -1.2), (-1.8, 1.5), (0.3, -1.8), (1.8, 0.4), (1.2, 1.8), (-0.4, 1.9)]
L = 0.9
for (px, py) in pts:
    pz = f(px, py)
    gx, gy = grad(px, py)
    norm = np.hypot(gx, gy)
    ux, uy = (gx / norm) * L, (gy / norm) * L
    ax.plot([px], [py], [pz], "o", color="black", ms=4, zorder=10)
    ax.quiver(px, py, pz, ux, uy, 0,
              color="crimson", linewidth=2.4, arrow_length_ratio=0.3,
              zorder=10)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
ax.set_title(r"Vectores $\nabla f$ superpuestos sobre la superficie"
             "\n(apuntan en la dirección de máximo crecimiento de f)")
fig.tight_layout()
fig.savefig(savepath("fig04_gradiente_superficie_3d.png"), dpi=150)
print("OK ->", savepath("fig04_gradiente_superficie_3d.png"))
