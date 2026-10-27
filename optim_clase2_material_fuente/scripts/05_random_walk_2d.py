"""
RANDOM WALK básico en 2D para minimización.

Distinto de "grid search" (malla fija) y distinto de "random search /
Monte Carlo puro" (muestras independientes en todo el dominio en cada
paso): un random walk da un paso aleatorio A PARTIR DEL PUNTO ACTUAL,
así que el recorrido es una cadena donde cada punto depende del anterior.

Se muestran DOS variantes:
  (a) random walk "greedy": el paso solo se acepta si mejora f (si no,
      se descarta y se prueba otro paso aleatorio desde el mismo punto).
      Esto SÍ es un optimizador simple (aunque ineficiente).
  (b) random walk "puro": el paso SIEMPRE se acepta, mejore o no. Esto es
      pura difusión -- no converge a un mínimo, solo sirve para explorar
      (se muestra como contraste, y para motivar la Clase 3: si en vez
      de "siempre" o "solo si mejora" usamos una probabilidad tipo
      Metrópolis, llegamos a simulated annealing).

No existe una función de scipy que sea "random walk" tal cual (es
demasiado simple/ineficiente para uso general) -- como comparación
"empaquetada" más honesta se usa Powell, un método de scipy que también
es derivative-free y avanza por direcciones sin usar gradiente.
"""
import numpy as np
from scipy.optimize import minimize
from _util import savepath
from _landscape import f, POZOS


def random_walk_greedy(f, x0, step=1.0, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    fx = f(*x)
    trayectoria = [x.copy()]
    aceptados = 0
    for _ in range(n_iter):
        paso = rng.uniform(-step, step, size=2)
        x_new = x + paso
        f_new = f(*x_new)
        if f_new < fx:                 # solo se mueve si MEJORA
            x, fx = x_new, f_new
            trayectoria.append(x.copy())
            aceptados += 1
    return x, fx, np.array(trayectoria), aceptados / n_iter


def random_walk_puro(f, x0, step=1.0, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    trayectoria = [x.copy()]
    x_best, f_best = x.copy(), f(*x)
    for _ in range(n_iter):
        paso = rng.uniform(-step, step, size=2)
        x = x + paso                    # SIEMPRE se mueve, mejore o no
        trayectoria.append(x.copy())
        fv = f(*x)
        if fv < f_best:
            x_best, f_best = x.copy(), fv
    return x_best, f_best, np.array(trayectoria)


if __name__ == "__main__":
    x0 = np.array([8.0, -8.0])

    xg, fg, tray_g, tasa = random_walk_greedy(f, x0, step=1.5, n_iter=500, seed=0)
    print(f"[greedy] x*={xg.round(3)}  f*={fg:.4f}  "
          f"({len(tray_g)-1} pasos aceptados de 500, tasa={tasa*100:.1f}%)")

    xp, fp, tray_p = random_walk_puro(f, x0, step=0.8, n_iter=150, seed=0)
    print(f"[puro]   mejor visto x*={xp.round(3)}  f*={fp:.4f}  "
          f"(el punto FINAL de la caminata es {tray_p[-1].round(3)}, "
          f"f={f(*tray_p[-1]):.4f} -- no necesariamente el mejor)")

    # comparacion con scipy (Powell: tambien derivative-free)
    res = minimize(lambda p: f(p[0], p[1]), x0, method="Powell")
    print(f"scipy    Powell -> x*={res.x.round(3)}  f*={res.fun:.4f}")

    # -------- figura --------
    import matplotlib.pyplot as plt
    xs = np.linspace(-10, 10, 250)
    ys = np.linspace(-10, 10, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, tray, x_best, f_best, titulo in [
        (axes[0], tray_g, xg, fg, f"Random walk GREEDY (solo acepta mejoras)\n"
                                   f"{len(tray_g)-1} pasos aceptados, mejor f={fg:.2f}"),
        (axes[1], tray_p, xp, fp, f"Random walk PURO (acepta todo)\n"
                                   f"solo explora, mejor visto f={fp:.2f}"),
    ]:
        ax.contourf(X, Y, Z, levels=25, cmap="viridis")
        ax.plot(tray[:, 0], tray[:, 1], "-", color="white", lw=0.8, alpha=0.8)
        ax.plot(tray[:, 0], tray[:, 1], ".", color="crimson", ms=3)
        ax.plot(*x0, "s", color="cyan", ms=10, mec="black", label="inicio")
        ax.plot(*x_best, "*", color="gold", ms=18, mec="black", label="mejor visto")
        for _, cx, cy, _ in POZOS:
            ax.plot(cx, cy, "+", color="black", ms=8, mew=1.5)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(savepath("fig05_random_walk.png"), dpi=150)
    print("\nFigura:", savepath("fig05_random_walk.png"))
