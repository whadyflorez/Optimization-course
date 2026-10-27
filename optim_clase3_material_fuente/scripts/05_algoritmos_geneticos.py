"""
ALGORITMOS GENÉTICOS (Holland, 1975; Goldberg, 1989), a mano.
Versión de valores reales (no codificación binaria) por simplicidad.

Ingredientes:
  1. Población inicial: N puntos aleatorios en el dominio.
  2. Fitness: qué tan bueno es cada individuo (aquí, -f, porque
     minimizamos f y "fitness alto = mejor").
  3. Selección: torneo -- se eligen k individuos al azar y gana el mejor
     (más simple y robusto que la ruleta clásica).
  4. Cruce (crossover): combinación aritmética (blend) de dos padres:
         hijo = beta*padre1 + (1-beta)*padre2 ,   beta ~ U(0,1)
  5. Mutación: perturbación gaussiana con probabilidad p_mut, para
     mantener diversidad y explorar.
  6. Elitismo: los mejores "e" individuos pasan sin cambios a la
     siguiente generación (para no perder el mejor logro).

Se compara con scipy.optimize.differential_evolution, que es también
un algoritmo evolutivo poblacional (misma familia, variante distinta
de cruce/mutación).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from _util import savepath
from _landscape import f, POZOS

BOUNDS = np.array([[-10.0, 10.0], [-10.0, 10.0]])


def poblacion_inicial(n, rng):
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    return rng.uniform(lo, hi, size=(n, 2))


def seleccion_torneo(pob, fit, k, rng):
    idx = rng.integers(0, len(pob), size=k)
    ganador = idx[np.argmax(fit[idx])]
    return pob[ganador]


def cruce_aritmetico(p1, p2, rng):
    beta = rng.random()
    return beta * p1 + (1 - beta) * p2


def mutar(hijo, p_mut, sigma, rng):
    if rng.random() < p_mut:
        hijo = hijo + rng.normal(scale=sigma, size=2)
        hijo = np.clip(hijo, BOUNDS[:, 0], BOUNDS[:, 1])
    return hijo


def algoritmo_genetico(n_pob=30, n_gen=60, k_torneo=3, p_mut=0.25,
                        sigma_mut=1.0, n_elite=2, seed=0):
    rng = np.random.default_rng(seed)
    pob = poblacion_inicial(n_pob, rng)
    fit = -f(pob[:, 0], pob[:, 1])

    mejor_hist = []
    mejor_x, mejor_f = None, np.inf

    for gen in range(n_gen):
        orden = np.argsort(-fit)
        pob, fit = pob[orden], fit[orden]

        if -fit[0] < mejor_f:
            mejor_f = -fit[0]
            mejor_x = pob[0].copy()
        mejor_hist.append(mejor_f)

        nueva_pob = [pob[i].copy() for i in range(n_elite)]  # elitismo
        while len(nueva_pob) < n_pob:
            p1 = seleccion_torneo(pob, fit, k_torneo, rng)
            p2 = seleccion_torneo(pob, fit, k_torneo, rng)
            hijo = cruce_aritmetico(p1, p2, rng)
            hijo = mutar(hijo, p_mut, sigma_mut, rng)
            nueva_pob.append(hijo)

        pob = np.array(nueva_pob)
        fit = -f(pob[:, 0], pob[:, 1])

    return mejor_x, mejor_f, np.array(mejor_hist), pob


if __name__ == "__main__":
    mejor_x, mejor_f, historia, pob_final = algoritmo_genetico(
        n_pob=30, n_gen=60, k_torneo=3, p_mut=0.25, sigma_mut=1.0,
        n_elite=2, seed=0)
    print(f"Manual  -> x* = {mejor_x.round(4)}   f(x*) = {mejor_f:.4f}   "
          f"(30 individuos x 60 generaciones)")

    # ======================================================================
    # Comparación con SCIPY
    # ======================================================================
    bounds = [(-10, 10), (-10, 10)]
    res = differential_evolution(lambda p: f(p[0], p[1]), bounds, seed=0,
                                  popsize=15, maxiter=60, tol=1e-8)
    print(f"scipy   -> x* = {res.x.round(4)}   f(x*) = {res.fun:.4f}   "
          f"(scipy.optimize.differential_evolution, misma familia evolutiva)")

    # ======================================================================
    # Figuras: población final sobre el paisaje + curva de convergencia
    # ======================================================================
    xs = np.linspace(-10, 10, 250)
    ys = np.linspace(-10, 10, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = f(X, Y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cs = ax1.contourf(X, Y, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax1, label="f(x,y)")
    ax1.scatter(pob_final[:, 0], pob_final[:, 1], c="white", edgecolors="black",
                s=35, label="población final", zorder=3)
    ax1.plot(*mejor_x, "*", color="gold", ms=22, mec="black", label="mejor encontrado")
    for _, cx, cy, _ in POZOS:
        ax1.plot(cx, cy, "+", color="red", ms=8, mew=1.5)
    mx_txt = tuple(float(v) for v in mejor_x.round(2))
    ax1.set_title(f"AG manual: población final, x*={mx_txt}, f={mejor_f:.3f}")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_aspect("equal")

    ax2.plot(historia, color="darkgreen", lw=2)
    ax2.set_xlabel("generación")
    ax2.set_ylabel("mejor f encontrado")
    ax2.set_title("Convergencia del algoritmo genético")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(savepath("fig05_algoritmo_genetico.png"), dpi=150)
    print("\nFigura:", savepath("fig05_algoritmo_genetico.png"))
