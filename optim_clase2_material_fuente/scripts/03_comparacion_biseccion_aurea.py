"""
Compara, en una sola figura, la velocidad de convergencia de bisección
(sobre f') contra búsqueda áurea (derivative-free), sobre el mismo
problema 1D: f(x) = x^2 - 2x + 3 en [-2, 5].
"""
import matplotlib.pyplot as plt
from importlib import import_module
from _util import savepath

biseccion = import_module("01_biseccion_1d")
aurea = import_module("02_busqueda_aurea_1d")

a0, b0 = -2.0, 5.0
_, _, hist_bis = biseccion.biseccion_min(biseccion.fprime, a0, b0, tol=1e-6, verbose=False)
_, _, hist_aur = aurea.busqueda_aurea(aurea.f, a0, b0, tol=1e-6, verbose=False)

anchos_bis = [b - a for (_, a, b, _, _) in hist_bis]
anchos_aur = [b - a for (_, a, b, _, _, _, _) in hist_aur]

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(range(1, len(anchos_bis) + 1), anchos_bis, "o-", color="#1f77b4",
            label=f"Bisección sobre $f'$  ({len(anchos_bis)} iter., "
                  f"~{2*len(anchos_bis)} evals de $f'$)")
ax.semilogy(range(1, len(anchos_aur) + 1), anchos_aur, "s-", color="#d62728",
            label=f"Búsqueda áurea  ({len(anchos_aur)} iter., "
                  f"~{len(anchos_aur)+1} evals de $f$)")
ax.set_xlabel("iteración")
ax.set_ylabel("ancho del intervalo (escala log)")
ax.set_title("Bisección vs. búsqueda áurea: misma tolerancia, mismo intervalo inicial")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(savepath("fig03_comparacion_biseccion_aurea.png"), dpi=150)
print("Figura:", savepath("fig03_comparacion_biseccion_aurea.png"))
print(f"\nBisección : {len(anchos_bis)} iteraciones para llegar a ancho {anchos_bis[-1]:.2e}")
print(f"Áurea     : {len(anchos_aur)} iteraciones para llegar a ancho {anchos_aur[-1]:.2e}")
