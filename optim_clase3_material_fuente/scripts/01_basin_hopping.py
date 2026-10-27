"""
BASIN HOPPING (Wales & Doye, 1997), implementado a mano.

Recordatorio del algoritmo (visto en detalle en clase):
  1. Partir de un mínimo local actual x_cur (optimizador local).
  2. Perturbar x_cur con un salto aleatorio -> x_prop.
  3. Reoptimizar localmente desde x_prop -> x_new, f_new.
  4. Aceptar con criterio de Metrópolis: siempre si f_new < f_cur; si no,
     con probabilidad P = exp(-(f_new - f_cur)/T).
  5. Llevar aparte el mejor (x_best, f_best) visto en TODA la corrida.

Función de ejemplo: el mismo paisaje de 6 pozos de la Clase 2
(_landscape.py) -- así los estudiantes reconocen el paisaje donde
random walk y Powell (Clase 2) se quedaron atrapados.
"""
import numpy as np
from scipy.optimize import basinhopping, minimize
from _util import savepath
from _landscape import f, POZOS


def grad_f(x, y, h=1e-6):
    gx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    gy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return np.array([gx, gy])


def local_search(x0, lr=0.1, max_iter=2000, tol=1e-8):
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        g = grad_f(*x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
    return x, float(f(*x))


def basin_hopping_manual(x0, n_iter=40, step_size=7.0, T=1.5, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    x_cur, f_cur = local_search(x0)
    x_best, f_best = x_cur.copy(), f_cur
    historia = [x_cur.copy()]

    for it in range(1, n_iter + 1):
        x_prop = x_cur + rng.uniform(-step_size, step_size, size=2)
        x_new, f_new = local_search(x_prop)

        if f_new < f_cur:
            aceptado = True
        else:
            aceptado = rng.random() < np.exp(-(f_new - f_cur) / T)

        if verbose and it <= 10:
            print(f"it={it:2d}  x_new={x_new.round(2)}  f_new={f_new: .4f}  "
                  f"f_cur={f_cur: .4f}  -> {'ACEPTADO' if aceptado else 'rechazado'}")

        if aceptado:
            x_cur, f_cur = x_new, f_new
        if f_new < f_best:
            x_best, f_best = x_new.copy(), f_new
        historia.append(x_new.copy() if aceptado else x_cur.copy())

    return x_best, f_best, np.array(historia)


if __name__ == "__main__":
    x0 = np.array([8.0, -8.0])

    x_best, f_best, historia = basin_hopping_manual(x0, n_iter=40, step_size=7.0, T=1.5, seed=0)
    print(f"\nManual  -> x* = {x_best.round(4)}   f(x*) = {f_best:.4f}   "
          f"({len(historia)-1} saltos)")

    # ======================================================================
    # Comparación con SCIPY
    # ======================================================================
    minimizer_kwargs = {"method": "BFGS"}
    res = basinhopping(lambda p: f(p[0], p[1]), x0, niter=40,
                        minimizer_kwargs=minimizer_kwargs, stepsize=7.0, T=1.5, seed=0)
    print(f"scipy   -> x* = {res.x.round(4)}   f(x*) = {res.fun:.4f}   "
          f"(scipy.optimize.basinhopping, niter=40)")

    # ======================================================================
    # Figura: trayectoria sobre el paisaje
    # ======================================================================
    import matplotlib.pyplot as plt
    xs = np.linspace(-10, 10, 250)
    ys = np.linspace(-10, 10, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, ax = plt.subplots(figsize=(7, 6.3))
    cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, label="f(x,y)")
    ax.plot(historia[:, 0], historia[:, 1], "o-", color="white", ms=5, lw=1.4,
            mfc="crimson", mec="white", label="cadena aceptada")
    ax.plot(*x0, "s", color="cyan", ms=11, mec="black", label="inicio")
    ax.plot(*x_best, "*", color="gold", ms=22, mec="black", label="mejor encontrado")
    for _, cx, cy, _ in POZOS:
        ax.plot(cx, cy, "+", color="black", ms=8, mew=1.5)
    xb_txt = tuple(float(v) for v in x_best.round(2))
    ax.set_title(f"Basin hopping manual: x*={xb_txt}, f={f_best:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(savepath("fig01_basin_hopping.png"), dpi=150)
    print("\nFigura:", savepath("fig01_basin_hopping.png"))
