"""
BASIN HOPPING EN 1D — versión reducida del script 01 para que los
estudiantes vean el mecanismo (salto + reoptimización local + Metrópolis)
en el caso más simple posible: una sola variable, se puede graficar todo
sobre la curva f(x).

Arranca en x0=8.5, muy cerca del pozo más somero (x~7.7, f~-0.83). Un
descenso de gradiente puro se queda atrapado ahí. Se necesita un salto
"hacia atrás" (aceptando temporalmente peor) para llegar al pozo global
en x*~-6.41.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath
from _landscape_1d import f, POZOS_1D


def fprime(x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def local_search(x0, lr=0.3, max_iter=2000, tol=1e-8):
    x = float(x0)
    for _ in range(max_iter):
        g = fprime(x)
        if abs(g) < tol:
            break
        x = x - lr * g
    return x, float(f(x))


def basin_hopping_1d(x0, n_iter=30, step_size=6.0, T=1.2, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    x_cur, f_cur = local_search(x0)
    x_best, f_best = x_cur, f_cur
    historia = [x_cur]

    for it in range(1, n_iter + 1):
        x_prop = x_cur + rng.uniform(-step_size, step_size)
        x_new, f_new = local_search(x_prop)

        if f_new < f_cur:
            aceptado = True
        else:
            aceptado = rng.random() < np.exp(-(f_new - f_cur) / T)

        if verbose and it <= 12:
            print(f"it={it:2d}  x_new={x_new: .3f}  f_new={f_new: .4f}  "
                  f"f_cur={f_cur: .4f}  -> {'ACEPTADO' if aceptado else 'rechazado'}")

        if aceptado:
            x_cur, f_cur = x_new, f_new
        if f_new < f_best:
            x_best, f_best = x_new, f_new
        historia.append(x_new if aceptado else x_cur)

    return x_best, f_best, np.array(historia)


if __name__ == "__main__":
    x0 = 8.5
    x_best, f_best, historia = basin_hopping_1d(x0, n_iter=30, step_size=9.0, T=1.5, seed=0)
    print(f"\nManual -> x* = {x_best:.4f}   f(x*) = {f_best:.4f}   ({len(historia) - 1} saltos)")

    # ======================================================================
    # Figura: curva f(x) + trayectoria de mínimos locales aceptados
    # ======================================================================
    xs = np.linspace(-10, 10, 800)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(xs, f(xs), color="steelblue", lw=2, zorder=1, label="$f(x)$")
    for _, c, _ in POZOS_1D:
        ax.axvline(c, color="black", lw=0.5, ls=":", alpha=0.4)

    ax.plot(historia, f(historia), "o-", color="white", mfc="crimson", mec="black",
            ms=7, lw=1.3, zorder=2, label="cadena aceptada")
    ax.plot(x0, f(x0), "s", color="cyan", ms=12, mec="black", zorder=3, label="inicio")
    ax.plot(x_best, f_best, "*", color="gold", ms=22, mec="black", zorder=3,
            label="mejor encontrado")

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Basin hopping 1D manual: x*={x_best:.3f}, f={f_best:.3f}")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(savepath("fig07_basin_hopping_1d.png"), dpi=150)
    print("Figura:", savepath("fig07_basin_hopping_1d.png"))
