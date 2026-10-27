"""
Concepto: clasificación de puntos críticos con la matriz Hessiana.
Tres superficies con un único punto crítico en el origen:
  - mínimo:  f1(x,y) =  x^2 + y^2         Hessiana definida positiva
  - máximo:  f2(x,y) = -x^2 - y^2         Hessiana definida negativa
  - silla :  f3(x,y) =  x^2 - y^2         Hessiana indefinida
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

x = np.linspace(-2, 2, 80)
y = np.linspace(-2, 2, 80)
X, Y = np.meshgrid(x, y)

funcs = [
    (r"Mínimo: $f=x^2+y^2$" + "\nHessiana definida positiva", X**2 + Y**2),
    (r"Máximo: $f=-x^2-y^2$" + "\nHessiana definida negativa", -X**2 - Y**2),
    (r"Silla: $f=x^2-y^2$" + "\nHessiana indefinida", X**2 - Y**2),
]

fig = plt.figure(figsize=(13, 4.5))
for i, (title, Z) in enumerate(funcs, start=1):
    ax = fig.add_subplot(1, 3, i, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="coolwarm", alpha=0.9, linewidth=0,
                     antialiased=True, rstride=2, cstride=2)
    ax.plot([0], [0], [0], "ko", ms=6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f")

fig.suptitle(r"Clasificación de puntos críticos según la Hessiana en $\nabla f=0$",
             fontsize=12)
fig.tight_layout()
fig.savefig(savepath("fig06_puntos_criticos.png"), dpi=150)
print("OK ->", savepath("fig06_puntos_criticos.png"))
