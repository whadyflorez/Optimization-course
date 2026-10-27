"""
PARTICLE SWARM OPTIMIZATION (Kennedy & Eberhart, 1995), a mano.

Idea: un enjambre de partículas "vuela" sobre el paisaje. Cada partícula
recuerda su mejor posición histórica (p_best) y "conoce" la mejor
posición hallada por todo el enjambre (g_best). Su velocidad se
actualiza combinando tres tendencias:

    v_new = w*v            (inercia: seguir en la misma dirección)
          + c1*r1*(p_best - x)   (memoria individual: volver a lo mejor propio)
          + c2*r2*(g_best - x)   (memoria social: ir hacia lo mejor del grupo)

    x_new = x + v_new

  w  (inercia)  grande -> explora más; chico -> converge más rápido
  c1 (cognitivo) qué tanto confía cada partícula en su propia experiencia
  c2 (social)    qué tanto confía en la experiencia del enjambre

No existe una función `scipy.optimize.pso`; la familia evolutiva/swarm
más cercana en scipy es, de nuevo, `differential_evolution` (también
poblacional, también sin gradiente). Para PSO "de verdad" en Python la
librería de referencia externa es `pyswarms` (no incluida en scipy).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from _util import savepath
from _landscape import f, POZOS

BOUNDS = np.array([[-10.0, 10.0], [-10.0, 10.0]])


def pso_manual(n_particulas=25, n_iter=60, w=0.7, c1=1.5, c2=1.5, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]

    x = rng.uniform(lo, hi, size=(n_particulas, 2))
    v = rng.uniform(-1, 1, size=(n_particulas, 2))

    p_best = x.copy()
    f_p_best = f(x[:, 0], x[:, 1])

    ig = np.argmin(f_p_best)
    g_best = p_best[ig].copy()
    f_g_best = f_p_best[ig]

    historia_gbest = [f_g_best]
    trayectoria = [x.copy()]

    for it in range(n_iter):
        r1 = rng.random((n_particulas, 2))
        r2 = rng.random((n_particulas, 2))
        v = (w * v
             + c1 * r1 * (p_best - x)
             + c2 * r2 * (g_best - x))
        x = np.clip(x + v, lo, hi)

        f_x = f(x[:, 0], x[:, 1])
        mejora = f_x < f_p_best
        p_best[mejora] = x[mejora]
        f_p_best[mejora] = f_x[mejora]

        ig = np.argmin(f_p_best)
        if f_p_best[ig] < f_g_best:
            f_g_best = f_p_best[ig]
            g_best = p_best[ig].copy()

        historia_gbest.append(f_g_best)
        trayectoria.append(x.copy())

    return g_best, f_g_best, np.array(historia_gbest), np.array(trayectoria)


if __name__ == "__main__":
    g_best, f_g_best, historia, trayectoria = pso_manual(
        n_particulas=25, n_iter=60, w=0.7, c1=1.5, c2=1.5, seed=0)
    print(f"Manual  -> x* = {g_best.round(4)}   f(x*) = {f_g_best:.4f}   "
          f"(25 partículas x 60 iteraciones)")

    # ======================================================================
    # Comparación con SCIPY (familia evolutiva más cercana: no hay PSO nativo)
    # ======================================================================
    bounds = [(-10, 10), (-10, 10)]
    res = differential_evolution(lambda p: f(p[0], p[1]), bounds, seed=0,
                                  popsize=15, maxiter=60, tol=1e-8)
    print(f"scipy   -> x* = {res.x.round(4)}   f(x*) = {res.fun:.4f}   "
          f"(scipy.optimize.differential_evolution; scipy no tiene PSO nativo,"
          f" pyswarms es la referencia externa)")

    # ======================================================================
    # Figuras: enjambre final sobre el paisaje + curva de convergencia
    # ======================================================================
    xs = np.linspace(-10, 10, 250)
    ys = np.linspace(-10, 10, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cs = ax1.contourf(X, Y, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label="f(x,y)")
    x_final = trayectoria[-1]
    ax1.scatter(x_final[:, 0], x_final[:, 1], c="white", edgecolors="black",
                s=35, label="enjambre final", zorder=3)
    ax1.plot(*g_best, "*", color="gold", ms=22, mec="black", label="g_best")
    for _, cx, cy, _ in POZOS:
        ax1.plot(cx, cy, "+", color="red", ms=8, mew=1.5)
    gb_txt = tuple(float(v) for v in g_best.round(2))
    ax1.set_title(f"PSO manual: enjambre final, x*={gb_txt}, f={f_g_best:.3f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_aspect("equal")

    ax2.plot(historia, color="navy", lw=2)
    ax2.set_xlabel("iteración")
    ax2.set_ylabel("mejor f del enjambre (g_best)")
    ax2.set_title("Convergencia de PSO")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath("fig06_particle_swarm.png"), dpi=150)
    print("\nFigura:", savepath("fig06_particle_swarm.png"))
