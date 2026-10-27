"""
Concepto: convexidad mediante la desigualdad de la cuerda.
f es convexa  <=>  para todo x1,x2 y t en [0,1]:
    f(t x1 + (1-t) x2)  <=  t f(x1) + (1-t) f(x2)
Panel izquierdo: función convexa (la cuerda queda SIEMPRE por encima).
Panel derecho: función NO convexa (la cuerda queda por debajo en tramos).
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

# --- convexa: f(x) = x^2 ---
fconv = lambda x: x**2
x = np.linspace(-2.5, 2.5, 300)
x1, x2 = -2.0, 1.7
t = np.linspace(0, 1, 50)
chord_x = t * x1 + (1 - t) * x2
chord_y = t * fconv(x1) + (1 - t) * fconv(x2)

ax1.plot(x, fconv(x), color="#1f77b4", lw=2.5, label=r"$f(x)=x^2$")
ax1.plot(chord_x, chord_y, "o-", color="crimson", ms=3, lw=1.8,
         label="cuerda entre $x_1,x_2$")
ax1.plot([x1, x2], [fconv(x1), fconv(x2)], "ko", ms=6)
ax1.set_title("Función convexa\n(la cuerda queda por encima de f)")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# --- no convexa: f(x) = x^4 - 4x^2 + x  (doble pozo asimétrico) ---
fnc = lambda x: x**4 - 4 * x**2 + x
x = np.linspace(-2.6, 2.4, 300)
x1, x2 = -1.9, 1.7
chord_y2 = t * fnc(x1) + (1 - t) * fnc(x2)

ax2.plot(x, fnc(x), color="#1f77b4", lw=2.5,
         label=r"$f(x)=x^4-4x^2+x$")
ax2.plot(chord_x * 0 + (t * x1 + (1 - t) * x2), chord_y2, "o-",
         color="crimson", ms=3, lw=1.8, label="cuerda entre $x_1,x_2$")
ax2.plot([x1, x2], [fnc(x1), fnc(x2)], "ko", ms=6)
ax2.set_title("Función NO convexa\n(la cuerda cruza por debajo de f)")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

for a in (ax1, ax2):
    a.set_xlabel("x"); a.set_ylabel("f(x)")

fig.tight_layout()
fig.savefig(savepath("fig08_convexidad.png"), dpi=150)
print("OK ->", savepath("fig08_convexidad.png"))
