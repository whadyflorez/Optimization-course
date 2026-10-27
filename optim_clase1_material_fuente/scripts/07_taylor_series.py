"""
Concepto: serie de Taylor como aproximación local de una función.
f(x) = sin(x) alrededor de x0 = 1 (radianes).
  P1(x) = f(x0) + f'(x0)(x-x0)                         (orden 1, recta)
  P2(x) = f(x0) + f'(x0)(x-x0) + 1/2 f''(x0)(x-x0)^2    (orden 2, parábola)
La aproximación de orden 2 es la que se usa en optimización (modelo
cuadrático local que usan métodos tipo Newton).
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

f = np.sin
fprime = np.cos
fsecond = lambda x: -np.sin(x)

x0 = 1.0
x = np.linspace(x0 - 2.5, x0 + 2.5, 400)

P1 = f(x0) + fprime(x0) * (x - x0)
P2 = P1 + 0.5 * fsecond(x0) * (x - x0) ** 2

fig, ax = plt.subplots(figsize=(6.8, 4.8))
ax.plot(x, f(x), color="black", lw=2.5, label=r"$f(x)=\sin(x)$")
ax.plot(x, P1, "--", color="#ff7f0e", lw=2,
        label=r"$P_1(x)$ (orden 1, tangente)")
ax.plot(x, P2, "-.", color="#2ca02c", lw=2,
        label=r"$P_2(x)$ (orden 2, cuadrática)")
ax.axvline(x0, color="gray", ls=":", lw=1)
ax.plot([x0], [f(x0)], "ko", ms=6)
ax.annotate(r"$x_0=1$", (x0, f(x0)), textcoords="offset points",
            xytext=(8, -18))
ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("Aproximación de Taylor de orden 1 y 2")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(savepath("fig07_taylor.png"), dpi=150)
print("OK ->", savepath("fig07_taylor.png"))
