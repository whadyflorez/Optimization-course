"""
PARTICLE SWARM OPTIMIZATION EN 1D — versión reducida del script 06. Con
una sola variable se puede graficar la trayectoria de CADA partícula a
lo largo de las iteraciones (posición vs. iteración), que es la forma
más clara de ver cómo el enjambre "converge en embudo" hacia g_best.

Se usa "inertia weight damping": w decrece linealmente de w_start a
w_end durante las iteraciones (mucha exploración al inicio, mucha
explotación al final -- el mismo espíritu que el enfriamiento de
simulated annealing). Sin este truco, w constante y alto produce
oscilaciones grandes que no se ven como un embudo claro en la gráfica.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath
from _landscape_1d import f, POZOS_1D

LO, HI = -10.0, 10.0


def pso_1d(n_particulas=20, n_iter=40, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)

    x = rng.uniform(LO, HI, size=n_particulas)
    v = rng.uniform(-1, 1, size=n_particulas)

    p_best = x.copy()
    f_p_best = f(x)

    ig = np.argmin(f_p_best)
    g_best = p_best[ig]
    f_g_best = f_p_best[ig]

    trayectoria = [x.copy()]
    historia_gbest = [f_g_best]

    for it in range(n_iter):
        w = w_start - (w_start - w_end) * it / n_iter        # damping lineal
        r1 = rng.random(n_particulas)
        r2 = rng.random(n_particulas)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = np.clip(x + v, LO, HI)

        f_x = f(x)
        mejora = f_x < f_p_best
        p_best[mejora] = x[mejora]
        f_p_best[mejora] = f_x[mejora]

        ig = np.argmin(f_p_best)
        if f_p_best[ig] < f_g_best:
            f_g_best = f_p_best[ig]
            g_best = p_best[ig]

        trayectoria.append(x.copy())
        historia_gbest.append(f_g_best)

    return g_best, f_g_best, np.array(trayectoria), np.array(historia_gbest)


if __name__ == "__main__":
    g_best, f_g_best, trayectoria, historia_gbest = pso_1d(
        n_particulas=20, n_iter=40, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5, seed=0)
    print(f"Manual -> x* = {g_best:.4f}   f(x*) = {f_g_best:.4f}   "
          f"(20 partículas x 40 iteraciones, w: 0.9 -> 0.4)")

    # ======================================================================
    # Figuras: trayectoria de cada partícula (posición vs iteración) +
    #          posiciones inicial/final sobre la curva f(x)
    # ======================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    n_iter = trayectoria.shape[0] - 1
    for i in range(trayectoria.shape[1]):
        ax1.plot(range(n_iter + 1), trayectoria[:, i], color="steelblue", alpha=0.5, lw=1.1)
    ax1.axhline(g_best, color="gold", lw=2, ls="--", label=f"g_best = {g_best:.2f}")
    for _, c, _ in POZOS_1D:
        ax1.axhline(c, color="black", lw=0.5, ls=":", alpha=0.3)
    ax1.set_xlabel("iteración")
    ax1.set_ylabel("posición x de cada partícula")
    ax1.set_title("PSO 1D: trayectorias del enjambre (embudo de convergencia)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    xs = np.linspace(-10, 10, 800)
    ax2.plot(xs, f(xs), color="steelblue", lw=2, zorder=1)
    for _, c, _ in POZOS_1D:
        ax2.axvline(c, color="black", lw=0.5, ls=":", alpha=0.4)
    x0_particulas = trayectoria[0]
    xf_particulas = trayectoria[-1]
    ax2.plot(x0_particulas, f(x0_particulas), "s", color="cyan", ms=8, mec="black",
              zorder=2, label="enjambre inicial", alpha=0.8)
    ax2.plot(xf_particulas, f(xf_particulas), "o", color="white", mec="black", ms=8,
              zorder=3, label="enjambre final")
    ax2.plot(g_best, f_g_best, "*", color="gold", ms=22, mec="black", zorder=4,
              label="g_best")
    ax2.set_xlabel("x"); ax2.set_ylabel("f(x)")
    ax2.set_title(f"Enjambre inicial vs. final: x*={g_best:.3f}, f={f_g_best:.3f}")
    ax2.legend(fontsize=9, loc="upper center")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(savepath("fig09_particle_swarm_1d.png"), dpi=150)
    print("Figura:", savepath("fig09_particle_swarm_1d.png"))
