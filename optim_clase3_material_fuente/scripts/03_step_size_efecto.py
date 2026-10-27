"""
Efecto de `step_size` (magnitud del salto aleatorio) en basin-hopping:
muy chico -> se queda atascado; muy grande -> pierde la ventaja de
"caminar" sobre cuencas vecinas (degenera a random-restart puro).
"""
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
from _util import savepath
from _landscape import f, POZOS

bh = import_module("01_basin_hopping")

X0 = np.array([8.0, -8.0])
T = 1.5
N_ITER = 40


def basin_hopping_medido(x0, step_size, n_iter=N_ITER, T=T, seed=0):
    rng = np.random.default_rng(seed)
    x_cur, f_cur = bh.local_search(x0)
    x_best, f_best = x_cur.copy(), f_cur
    historia = [x_cur.copy()]
    saltos_inutiles = 0
    for _ in range(n_iter):
        x_antes = x_cur.copy()
        x_prop = x_cur + rng.uniform(-step_size, step_size, size=2)
        x_new, f_new = bh.local_search(x_prop)
        if np.linalg.norm(x_new - x_antes) < 1e-2:
            saltos_inutiles += 1
        aceptado = f_new < f_cur or rng.random() < np.exp(-(f_new - f_cur) / T)
        if aceptado:
            x_cur, f_cur = x_new, f_new
        if f_new < f_best:
            x_best, f_best = x_new.copy(), f_new
        historia.append(x_new.copy())
    historia = np.array(historia)
    n_basins = len({tuple(p.round(1)) for p in historia})
    return x_best, f_best, historia, n_basins, saltos_inutiles / n_iter


casos = [0.5, 3.0, 7.0, 20.0]
resultados = {}
print(f"{'step_size':>10} {'f_best':>9} {'basins':>8} {'saltos inutiles':>16}")
for s in casos:
    x_best, f_best, historia, n_basins, frac = basin_hopping_medido(X0, s)
    resultados[s] = (x_best, f_best, historia, n_basins, frac)
    print(f"{s:10.1f} {f_best:9.3f} {n_basins:8d} {frac*100:15.1f}%")

xs = np.linspace(-10, 10, 220)
ys = np.linspace(-10, 10, 220)
Xg, Yg = np.meshgrid(xs, ys)
Zg = f(Xg, Yg)

fig, axes = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
for ax, s in zip(axes, casos):
    x_best, f_best, historia, n_basins, frac = resultados[s]
    ax.contourf(Xg, Yg, Zg, levels=25, cmap="viridis")
    ax.plot(historia[:, 0], historia[:, 1], "o-", color="white", ms=4, lw=1.2,
            mfc="crimson", mec="white")
    ax.plot(*X0, "s", color="cyan", ms=10, mec="black")
    ax.plot(*x_best, "*", color="gold", ms=18, mec="black")
    for _, cx, cy, _ in POZOS:
        ax.plot(cx, cy, "+", color="black", ms=8, mew=1.5)
    etiqueta = {0.5: "muy chico", 3.0: "chico", 7.0: "razonable", 20.0: "muy grande"}[s]
    ax.set_title(f"step_size={s} ({etiqueta})\n{n_basins} cuencas, "
                 f"{frac*100:.0f}% saltos inútiles\nmejor f={f_best:.2f}", fontsize=9.5)
    ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_aspect("equal")

fig.suptitle("Efecto de la magnitud del salto aleatorio (step_size)", y=1.03)
fig.tight_layout()
fig.savefig(savepath("fig03_step_size.png"), dpi=150, bbox_inches="tight")
print("\nFigura:", savepath("fig03_step_size.png"))
