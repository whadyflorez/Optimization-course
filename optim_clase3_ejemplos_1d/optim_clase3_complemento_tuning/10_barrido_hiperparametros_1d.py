"""
BARRIDO DE HIPERPARÁMETROS (T, step_size) para basin hopping 1D.

Este script es la evidencia detrás de la discusión "¿el ajuste de T y
step_size es solo tanteo?": en vez de correr una sola vez y quedarnos
con lo que salga, medimos la TASA DE ÉXITO (¿encontró el mínimo global?)
sobre muchas semillas distintas para cada combinación de hiperparámetros.
Esto es tuning sistemático, no cherry-picking de una semilla con suerte.

Nota importante (el punto que motiva la sección "Complemento" de la
Clase 3): esto solo es posible porque en este ejemplo de clase
CONOCEMOS el mínimo global de antemano (x*=-6.41, f*=-4.166) y podemos
usarlo como vara de medir. En un problema real no se tiene esa
referencia -- ver la discusión de diagnósticos alternativos en las
slides.
"""
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
from _util import savepath

bh = import_module("07_basin_hopping_1d")

X0 = 8.5
N_SEEDS = 30
F_GLOBAL = -4.0   # umbral: cualquier resultado mas bajo que esto se cuenta como "encontro el global"

STEPS = [3, 5, 7, 9, 11, 13]
TEMPS = [0.5, 1.0, 1.5, 2.5, 4.0]


def tasa_exito(step_size, T, n_seeds=N_SEEDS):
    hits = 0
    for seed in range(n_seeds):
        _, f_best, _ = bh.basin_hopping_1d(X0, n_iter=30, step_size=step_size, T=T,
                                            seed=seed, verbose=False)
        if f_best < F_GLOBAL:
            hits += 1
    return 100.0 * hits / n_seeds


if __name__ == "__main__":
    grid = np.zeros((len(TEMPS), len(STEPS)))
    print(f"{'step_size':>10}", *[f"T={t}" for t in TEMPS])
    for j, s in enumerate(STEPS):
        fila = []
        for i, T in enumerate(TEMPS):
            tasa = tasa_exito(s, T)
            grid[i, j] = tasa
            fila.append(tasa)
        print(f"{s:10.1f}", *[f"{v:5.0f}%" for v in fila])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto", origin="lower")
    ax.set_xticks(range(len(STEPS))); ax.set_xticklabels(STEPS)
    ax.set_yticks(range(len(TEMPS))); ax.set_yticklabels(TEMPS)
    ax.set_xlabel("step_size")
    ax.set_ylabel("T")
    ax.set_title(f"Basin hopping 1D: % de {N_SEEDS} semillas que encuentran el global")
    for i in range(len(TEMPS)):
        for j in range(len(STEPS)):
            v = grid[i, j]
            color = "black" if v > 45 else "white"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color=color, fontsize=10)
    fig.colorbar(im, ax=ax, label="tasa de éxito (%)")
    fig.tight_layout()
    fig.savefig(savepath("fig10_barrido_hiperparametros.png"), dpi=150)
    print("\nFigura:", savepath("fig10_barrido_hiperparametros.png"))
