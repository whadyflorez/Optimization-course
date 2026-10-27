"""
SIMULATED ANNEALING EN 1D — versión reducida del script 04. Igual que en
2D: se perturba el punto actual y se acepta/rechaza directamente con
Metrópolis, SIN reoptimizar localmente en cada paso (a diferencia de
basin hopping). El enfriamiento geométrico T_k = T0*alpha^k concentra la
búsqueda cerca del mejor punto hallado a medida que avanzan las
iteraciones.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath
from _landscape_1d import f, POZOS_1D


def simulated_annealing_1d(x0, T0=6.0, alpha=0.96, n_iter=150, paso=1.2, seed=0):
    rng = np.random.default_rng(seed)
    x_cur = float(x0)
    f_cur = float(f(x_cur))
    x_best, f_best = x_cur, f_cur
    historia = [x_cur]
    T_historia = []
    T = T0

    for it in range(n_iter):
        x_prop = x_cur + rng.normal(scale=paso)
        f_prop = float(f(x_prop))
        dE = f_prop - f_cur

        aceptado = dE < 0 or rng.random() < np.exp(-dE / T)
        if aceptado:
            x_cur, f_cur = x_prop, f_prop
        if f_cur < f_best:
            x_best, f_best = x_cur, f_cur

        historia.append(x_cur)
        T_historia.append(T)
        T *= alpha

    return x_best, f_best, np.array(historia), np.array(T_historia)


if __name__ == "__main__":
    x0 = 8.5
    x_best, f_best, historia, T_hist = simulated_annealing_1d(
        x0, T0=6.0, alpha=0.97, n_iter=150, paso=3.5, seed=0)
    print(f"Manual -> x* = {x_best:.4f}   f(x*) = {f_best:.4f}   "
          f"({len(historia) - 1} pasos, T: {T_hist[0]:.2f} -> {T_hist[-1]:.4f})")

    # ======================================================================
    # Figuras: trayectoria sobre la curva (coloreada por iteración) + enfriamiento
    # ======================================================================
    xs = np.linspace(-10, 10, 800)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(xs, f(xs), color="steelblue", lw=2, zorder=1)
    for _, c, _ in POZOS_1D:
        ax1.axvline(c, color="black", lw=0.5, ls=":", alpha=0.4)
    ax1.plot(historia, f(historia), "-", color="lightgray", lw=0.8, zorder=2)
    sc = ax1.scatter(historia, f(historia), c=np.arange(len(historia)), cmap="autumn",
                      s=22, zorder=3, label="cadena")
    ax1.plot(x0, f(x0), "s", color="cyan", ms=12, mec="black", zorder=4, label="inicio")
    ax1.plot(x_best, f_best, "*", color="gold", ms=22, mec="black", zorder=4,
              label="mejor encontrado")
    ax1.set_xlabel("x"); ax1.set_ylabel("f(x)")
    ax1.set_title(f"Simulated annealing 1D: x*={x_best:.3f}, f={f_best:.3f}")
    ax1.legend(fontsize=9, loc="upper center")
    ax1.grid(alpha=0.25)

    ax2.plot(T_hist, color="firebrick", lw=2)
    ax2.set_xlabel("iteración")
    ax2.set_ylabel("T (temperatura)")
    ax2.set_title(r"Enfriamiento geométrico: $T_k = T_0\,\alpha^k$")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath("fig08_simulated_annealing_1d.png"), dpi=150)
    print("Figura:", savepath("fig08_simulated_annealing_1d.png"))
