"""
Búsqueda EXHAUSTIVA tipo grid (grid search) en 1, 2 y 3 variables.

Idea: evaluar f en TODOS los puntos de una malla regular y quedarse con
el mejor. No requiere derivadas, no requiere que f sea continua, no le
importa cuántos mínimos locales tenga -- siempre encuentra el mínimo de
la malla (que se acerca al mínimo real si la malla es suficientemente
fina). El precio: el número de evaluaciones crece como N^d (N puntos por
eje, d variables) -- la "maldición de la dimensionalidad".
"""
import itertools
import numpy as np
from scipy.optimize import brute
from _util import savepath
from _landscape import f as f2d, POZOS

# ======================================================================
# 1) Grid search en 1 variable:  f(x) = x^2 - 2x + 3  en [-2,5]
# ======================================================================
f1d = lambda x: x**2 - 2 * x + 3
N = 50
xs = np.linspace(-2, 5, N)
vals = f1d(xs)
i_min = np.argmin(vals)
print(f"[1D] N={N} puntos -> {N} evaluaciones -> x*~{xs[i_min]:.4f}  f*~{vals[i_min]:.4f}")

# ======================================================================
# 2) Grid search en 2 variables: el paisaje de 6 pozos (_landscape.py)
# ======================================================================
N_ = [10,25,50,100]
for N in N_:
      xs2 = np.linspace(-10, 10, N)
      ys2 = np.linspace(-10, 10, N)
      X2, Y2 = np.meshgrid(xs2, ys2)
      Z2 = f2d(X2, Y2)
      idx = np.unravel_index(np.argmin(Z2), Z2.shape)
      x2_star, y2_star = X2[idx], Y2[idx]
      print(f"[2D] N={N} puntos/eje -> {N**2} evaluaciones -> "
            f"(x*,y*)~({x2_star:.3f},{y2_star:.3f})  f*~{Z2[idx]:.4f}")

# ======================================================================
# 3) Grid search en 3 variables: paraboloide simple f(x,y,z)
# ======================================================================
f3d = lambda x, y, z: (x - 1)**2 + (y + 1)**2 + (z - 2)**2
N3 = 40   # mas chico: N3^3 evaluaciones ya es bastante
xs3 = np.linspace(-4, 4, N3)
ys3 = np.linspace(-4, 4, N3)
zs3 = np.linspace(-4, 4, N3)

mejor = None
for x, y, z in itertools.product(xs3, ys3, zs3):   # equivalente a 3 for anidados
    v = f3d(x, y, z)
    if mejor is None or v < mejor[0]:
        mejor = (v, x, y, z)
print(f"[3D] N={N3} puntos/eje -> {N3**3} evaluaciones -> "
      f"(x*,y*,z*)~({mejor[1]:.3f},{mejor[2]:.3f},{mejor[3]:.3f})  f*~{mejor[0]:.4f}")

# ======================================================================
# 4) La maldición de la dimensionalidad: costo N^d
# ======================================================================
print("\nCosto (evaluaciones) para N=50 puntos por eje, variando d:")
for d in range(1, 6):
    print(f"  d={d}  ->  50^{d} = {50**d:,} evaluaciones")

# ======================================================================
# 5) Comparación con SCIPY (caso 2D): scipy.optimize.brute
# ======================================================================
rango = (slice(-10, 10, 20 / N), slice(-10, 10, 20 / N))  # misma resolucion, N pasos
res = brute(lambda p: f2d(p[0], p[1]), rango, finish=None)  # finish=None: SOLO grid, sin refinar
print(f"\n[2D] scipy.optimize.brute (finish=None, grid puro) -> "
      f"(x*,y*)=({res[0]:.3f},{res[1]:.3f})  f*={f2d(*res):.4f}")

res_refinado = brute(lambda p: f2d(p[0], p[1]), rango, finish=None)
res_con_refinamiento = brute(lambda p: f2d(p[0], p[1]), rango)  # finish=fmin (default): pule con Nelder-Mead
print(f"[2D] scipy.optimize.brute (con refinamiento local, default) -> "
      f"(x*,y*)=({res_con_refinamiento[0]:.3f},{res_con_refinamiento[1]:.3f})  "
      f"f*={f2d(*res_con_refinamiento):.4f}")

# ======================================================================
# 6) Figuras
# ======================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# -- panel izquierdo: grid 2D con el punto encontrado --
cs = axes[0].contourf(X2, Y2, Z2, levels=30, cmap="viridis")
fig.colorbar(cs, ax=axes[0], label="f(x,y)")
axes[0].scatter(X2, Y2, s=1, color="white", alpha=0.15)  # la malla evaluada
axes[0].plot(x2_star, y2_star, "*", color="gold", ms=20, mec="black",
             label=f"mínimo de la malla\n({x2_star:.2f},{y2_star:.2f})")
for _, cx, cy, _ in POZOS:
    axes[0].plot(cx, cy, "+", color="black", ms=8, mew=1.5)
axes[0].set_title(f"Grid search 2D: {N}x{N}={N**2} evaluaciones")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].legend(fontsize=8, loc="lower left")
axes[0].set_aspect("equal")

# -- panel derecho: costo N^d --
d_vals = np.arange(1, 7)
axes[1].semilogy(d_vals, 50**d_vals, "o-", color="crimson", ms=8)
axes[1].set_xlabel("número de variables (d)")
axes[1].set_ylabel("evaluaciones de f (escala log)")
axes[1].set_title("Costo del grid search: $N^d$ con $N=50$ puntos/eje")
axes[1].grid(alpha=0.3, which="both")
for d, val in zip(d_vals, 50**d_vals):
    axes[1].annotate(f"{val:,.0f}", (d, val), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(savepath("fig04_grid_search.png"), dpi=150)
print("\nFigura:", savepath("fig04_grid_search.png"))
