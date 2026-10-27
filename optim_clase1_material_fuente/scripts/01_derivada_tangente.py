"""
Concepto: derivada ordinaria como pendiente de la recta tangente.
f(x) = x^2 - 2x + 3 ;  f'(x) = 2x - 2
Se grafica f, la recta secante (aproximacion con incremento h) y la recta
tangente en x0, para visualizar f'(x0) = lim_{h->0} [f(x0+h) - f(x0)] / h.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = lambda x: x**2 - 2 * x + 3
fprime = lambda x: 2 * x - 2

x0 = 2.0
h = 1.2  # incremento "grande" solo para dibujar la secante

x = np.linspace(-1, 5, 400)
y = f(x)

m_sec = (f(x0 + h) - f(x0)) / h          # pendiente secante
m_tan = fprime(x0)                        # pendiente tangente exacta

recta_sec = f(x0) + m_sec * (x - x0)
recta_tan = f(x0) + m_tan * (x - x0)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(x, y, color="#1f77b4", lw=2.5, label=r"$f(x) = x^2 - 2x + 3$")
ax.plot(x, recta_sec, "--", color="#ff7f0e", lw=1.8,
        label=fr"secante ($h={h}$), pendiente $\approx {m_sec:.2f}$")
ax.plot(x, recta_tan, color="#2ca02c", lw=2.2,
        label=fr"tangente en $x_0$, pendiente $f'(x_0)={m_tan:.2f}$")
ax.plot([x0, x0 + h], [f(x0), f(x0 + h)], "o", color="#ff7f0e", zorder=5)
ax.plot([x0], [f(x0)], "o", color="#2ca02c", ms=8, zorder=6)
ax.annotate(r"$x_0$", (x0, f(x0)), textcoords="offset points",
            xytext=(-15, -20), fontsize=12)

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Derivada como límite de la pendiente de la secante")
ax.legend(loc="upper center", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(savepath("fig01_derivada_tangente.png"))
print("OK ->", savepath("fig01_derivada_tangente.png"))
