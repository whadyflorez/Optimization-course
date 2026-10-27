"""
SIMULATED ANNEALING (Kirkpatrick, Gelatt & Vecchi, 1983), a mano.

Diferencia clave con basin-hopping:
  - Basin hopping perturba y luego SIEMPRE reoptimiza localmente
    (gradient descent) antes de comparar/aceptar.
  - Simulated annealing perturba el punto actual y compara/acepta
    DIRECTAMENTE con el criterio de Metrópolis, SIN minimizar
    localmente en cada paso. El "aterrizaje suave" en el mínimo ocurre
    solo al final, cuando T ya es muy chica.

Enlace con la Clase 2 (unifica el marco de random walk):
  - random walk "greedy" (Clase 2)  ==  SA con T = 0   (nunca acepta empeorar)
  - random walk "puro"   (Clase 2)  ==  SA con T = inf (siempre acepta)
  - SA real = un puente entre los dos, con T decreciendo con el tiempo
    (annealing / templado): explora como random walk puro al principio,
    converge como random walk greedy al final.

Esquema de enfriamiento (geométrico, el más usado en la práctica):
    T_k = T_0 * alpha^k ,   0 < alpha < 1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing
from _util import savepath
from _landscape import f, POZOS


def simulated_annealing_manual(x0, T0=8.0, alpha=0.95, n_iter=150,
                                paso=1.5, seed=0):
    rng = np.random.default_rng(seed)
    x_cur = np.array(x0, dtype=float)
    f_cur = float(f(*x_cur))
    x_best, f_best = x_cur.copy(), f_cur
    historia = [x_cur.copy()]
    T_historia = []
    T = T0

    for it in range(n_iter):
        x_prop = x_cur + rng.normal(scale=paso, size=2)
        f_prop = float(f(*x_prop))
        dE = f_prop - f_cur

        if dE < 0:
            aceptado = True
        else:
            aceptado = rng.random() < np.exp(-dE / T)

        if aceptado:
            x_cur, f_cur = x_prop, f_prop
        if f_cur < f_best:
            x_best, f_best = x_cur.copy(), f_cur

        historia.append(x_cur.copy())
        T_historia.append(T)
        T *= alpha  # enfriamiento geometrico

    return x_best, f_best, np.array(historia), np.array(T_historia)


if __name__ == "__main__":
    x0 = np.array([8.0, -8.0])

    x_best, f_best, historia, T_hist = simulated_annealing_manual(
        x0, T0=8.0, alpha=0.96, n_iter=200, paso=1.5, seed=0)
    print(f"Manual  -> x* = {x_best.round(4)}   f(x*) = {f_best:.4f}   "
          f"({len(historia) - 1} pasos, T: {T_hist[0]:.2f} -> {T_hist[-1]:.4f})")

    # ======================================================================
    # Comparación con SCIPY
    # ======================================================================
    bounds = [(-10, 10), (-10, 10)]
    res = dual_annealing(lambda p: f(p[0], p[1]), bounds, seed=0, maxiter=200)
    print(f"scipy   -> x* = {res.x.round(4)}   f(x*) = {res.fun:.4f}   "
          f"(scipy.optimize.dual_annealing)")

    # ======================================================================
    # Figuras: trayectoria + esquema de enfriamiento
    # ======================================================================
    xs = np.linspace(-10, 10, 250)
    ys = np.linspace(-10, 10, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cs = ax1.contourf(X, Y, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label="f(x,y)")
    ax1.plot(historia[:, 0], historia[:, 1], "-", color="white", lw=0.9, alpha=0.85)
    ax1.scatter(historia[:, 0], historia[:, 1], c=np.arange(len(historia)),
                cmap="autumn", s=10, zorder=3)
    ax1.plot(*x0, "s", color="cyan", ms=11, mec="black", label="inicio")
    ax1.plot(*x_best, "*", color="gold", ms=22, mec="black", label="mejor encontrado")
    for _, cx, cy, _ in POZOS:
        ax1.plot(cx, cy, "+", color="black", ms=8, mew=1.5)
    xb_txt = tuple(float(v) for v in x_best.round(2))
    ax1.set_title(f"Simulated annealing manual: x*={xb_txt}, f={f_best:.3f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.set_xlim(-10, 10); ax1.set_ylim(-10, 10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_aspect("equal")

    ax2.plot(T_hist, color="firebrick", lw=2)
    ax2.set_xlabel("iteración")
    ax2.set_ylabel("T (temperatura)")
    ax2.set_title(r"Esquema de enfriamiento geométrico: $T_k = T_0\,\alpha^k$")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath("fig04_simulated_annealing.png"), dpi=150)
    print("\nFigura:", savepath("fig04_simulated_annealing.png"))
