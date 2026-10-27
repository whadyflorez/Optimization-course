"""
Minimización 1D por BISECCIÓN.

Idea: un mínimo de f (si f es diferenciable) es una raíz de f'(x)=0. Si
tenemos un intervalo [a,b] donde f'(a) y f'(b) tienen signos opuestos
(f decreciente en a, creciente en b -> hay un mínimo adentro), podemos
usar bisección clásica sobre f' para encontrar esa raíz.

Requiere: calcular f' (analítica o numérica) y un intervalo inicial
donde el signo de f' cambie. Convergencia LINEAL: el intervalo se
reduce a la mitad en cada iteración, sin importar la forma de f.

Función de ejemplo: la misma de la Clase 1, f(x) = x^2 - 2x + 3,
f'(x) = 2x - 2, mínimo exacto en x*=1.
"""
import numpy as np
from scipy.optimize import brentq
from _util import savepath

f = lambda x: x**2 - 2 * x + 3
fprime = lambda x: 2 * x - 2


# ======================================================================
# 1) Implementación MANUAL
# ======================================================================
def biseccion_min(fprime, a, b, tol=1e-6, max_iter=100, verbose=True):
    assert fprime(a) * fprime(b) < 0, "f'(a) y f'(b) deben tener signos opuestos"
    historia = []
    for it in range(1, max_iter + 1):
        c = 0.5 * (a + b)
        fc = fprime(c)
        historia.append((it, a, b, c, fc))
        if verbose and it <= 12:
            print(f"it={it:2d}  a={a: .6f}  b={b: .6f}  c={c: .6f}  "
                  f"f'(c)={fc: .6f}  ancho={b-a:.6f}")
        if abs(fc) < tol or (b - a) < tol:
            break
        if np.sign(fc) == np.sign(fprime(a)):
            a = c
        else:
            b = c
    return c, f(c), historia


if __name__ == "__main__":
    a0, b0 = -2.0, 5.0
    print(f"Intervalo inicial: [{a0},{b0}]   f'(a)={fprime(a0)}  f'(b)={fprime(b0)}")
    x_star, f_star, historia = biseccion_min(fprime, a0, b0, tol=1e-6)
    print(f"\nManual  -> x* = {x_star:.6f}   f(x*) = {f_star:.6f}  "
          f"({len(historia)} iteraciones)")

    # ======================================================================
    # 2) Comparación con SCIPY: brentq encuentra la raíz de f' de forma
    #    mucho más rápida (usa secante/interp. cuadrática inversa, no solo
    #    bisección pura), pero resuelve el MISMO problema matemático.
    # ======================================================================
    x_scipy = brentq(fprime, a0, b0, xtol=1e-6)
    print(f"scipy   -> x* = {x_scipy:.6f}   f(x*) = {f(x_scipy):.6f}   "
          f"(scipy.optimize.brentq sobre f')")

    # ======================================================================
    # 3) Figura: convergencia del ancho del intervalo (escala log)
    # ======================================================================
    import matplotlib.pyplot as plt
    anchos = [b - a for (_, a, b, _, _) in historia]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(range(1, len(anchos) + 1), anchos, "o-", color="#1f77b4",
                label="ancho del intervalo $b-a$")
    ax.set_xlabel("iteración")
    ax.set_ylabel(r"ancho del intervalo (escala log)")
    ax.set_title("Bisección: convergencia lineal (se reduce a la mitad c/iter.)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath("fig01_biseccion_convergencia.png"), dpi=150)
    print("\nFigura:", savepath("fig01_biseccion_convergencia.png"))
