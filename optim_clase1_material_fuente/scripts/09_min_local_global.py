"""
Concepto: mínimos/máximos locales vs. globales.
f(x) = x^4 - 4x^2 + x   (doble pozo asimétrico)
f'(x) = 4x^3 - 8x + 1  -> se resuelve numéricamente (raíces de un cúbico)
para ubicar los 3 puntos críticos y clasificarlos con f''(x)=12x^2-8.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x: x**4 - 4 * x**2 + x
fsecond = lambda x: 12 * x**2 - 8

# raices de f'(x) = 4x^3 - 8x + 1 = 0
roots = np.roots([4, 0, -8, 1])
roots = np.sort(roots[np.isreal(roots)].real)

x = np.linspace(-2.6, 2.4, 400)
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.plot(x, f(x), color="#1f77b4", lw=2.5)

labels = []
for r in roots:
    curv = fsecond(r)
    kind = "mínimo" if curv > 0 else "máximo"
    labels.append((r, f(r), kind))

# el mínimo con menor valor de f es el global
minimos = [(r, val) for r, val, k in labels if k == "mínimo"]
r_global = min(minimos, key=lambda p: p[1])[0]

for r, val, kind in labels:
    es_global = (kind == "mínimo" and np.isclose(r, r_global))
    color = "green" if es_global else ("crimson" if kind == "mínimo" else "orange")
    etiqueta = ("mínimo global" if es_global else
                ("mínimo local" if kind == "mínimo" else "máximo local"))
    ax.plot([r], [val], "o", color=color, ms=9, zorder=5)
    ax.annotate(f"{etiqueta}\n({r:.2f}, {val:.2f})", (r, val),
                textcoords="offset points", xytext=(0, 14 if kind == "máximo" else -35),
                ha="center", fontsize=9, color=color)

ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title(r"$f(x)=x^4-4x^2+x$: mínimos/máximo locales y global")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(savepath("fig09_min_local_global.png"), dpi=150)
print("OK -> puntos criticos:", labels)
