"""
EL PROBLEMA DEL SULTÁN / LA SECRETARIA -- verificación por Monte Carlo
de la regla del 1/e (37%).

Estrategia de umbral: rechazar automáticamente a las primeras k
candidatas (solo sirven para calibrar qué tan buena es "buena"), y
luego aceptar a la primera candidata siguiente que sea mejor que TODAS
las anteriores (un "récord"). Si nadie después de la posición k es un
récord, no se elige a nadie (fracaso).

Este script:
  1. Simula esa estrategia miles de veces para cada valor de k y mide
     la tasa de éxito (¿se quedó con la candidata realmente mejor de
     las n?).
  2. La compara contra la fórmula teórica EXACTA.
  3. Muestra que el óptimo empírico coincide con k/n ~ 1/e y que la
     tasa de éxito en el óptimo es ~1/e ~ 36.8%, sin importar n.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

N = 100          # numero de candidatas
TRIALS = 20000   # repeticiones Monte Carlo por cada k


def tasa_exito_simulada(k, n, trials, rng):
    # cada fila: una permutacion aleatoria de los rangos 1..n (1 = la mejor)
    perms = np.array([rng.permutation(n) + 1 for _ in range(trials)])

    if k == 0:
        mejor_visto = np.full(trials, np.inf)      # nada visto todavia -> se acepta la 1a
    else:
        mejor_visto = perms[:, :k].min(axis=1)      # el mejor rango entre las primeras k (rechazadas)

    candidatas = perms[:, k:]                        # las que SI se consideran para elegir
    es_record = candidatas < mejor_visto[:, None]     # mejor que todo lo visto antes
    hubo_eleccion = es_record.any(axis=1)
    idx_primero = np.argmax(es_record, axis=1)        # primer indice True (0 si no hay ninguno)
    elegida = candidatas[np.arange(trials), idx_primero]

    exito = hubo_eleccion & (elegida == 1)            # 1 = rango de la mejor candidata de todas
    return exito.mean()


def tasa_exito_teorica(k, n):
    if k == 0:
        return 1.0 / n
    return (k / n) * sum(1.0 / (i - 1) for i in range(k + 1, n + 1))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ks = np.arange(0, N)

    tasas_sim = np.array([tasa_exito_simulada(k, N, TRIALS, rng) for k in ks])
    tasas_teo = np.array([tasa_exito_teorica(k, N) for k in ks])

    k_opt_sim = ks[np.argmax(tasas_sim)]
    k_opt_teo = ks[np.argmax(tasas_teo)]

    print(f"n = {N} candidatas, {TRIALS} repeticiones Monte Carlo por cada k\n")
    print(f"k optimo (simulado):  {k_opt_sim:3d}   tasa de exito = {tasas_sim.max():.4f}")
    print(f"k optimo (teorico):   {k_opt_teo:3d}   tasa de exito = {tasas_teo.max():.4f}")
    print(f"n/e = {N/np.e:.2f}   (redondeado: {round(N/np.e)})")
    print(f"1/e = {1/np.e:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(ks / N, tasas_teo, color="steelblue", lw=2, zorder=2,
            label="teórica (fórmula exacta)")
    ax.plot(ks / N, tasas_sim, "o", color="crimson", ms=3, alpha=0.55, zorder=3,
            label=f"simulada Monte Carlo ({TRIALS} repet./k)")
    ax.axvline(1 / np.e, color="gray", ls="--", lw=1.2, zorder=1, label="k/n = 1/e")
    ax.axhline(1 / np.e, color="gray", ls=":", lw=1.2, zorder=1)
    ax.plot(k_opt_sim / N, tasas_sim.max(), "*", color="gold", ms=18, mec="black", zorder=4)
    ax.set_xlabel("k / n  (fracción de candidatas rechazada de entrada)")
    ax.set_ylabel("P(elegir a la mejor candidata de todas)")
    ax.set_title("Regla del 1/e: tasa de éxito vs. fracción rechazada")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath("fig_secretaria_1e.png"), dpi=150)
    print("\nFigura:", savepath("fig_secretaria_1e.png"))
