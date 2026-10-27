"""
Minimización 1D por BÚSQUEDA ÁUREA (golden section search).

Idea: a diferencia de bisección, NO necesita derivada -- solo evalúa f.
Sirve para cualquier función unimodal (un solo mínimo) en [a,b], incluso
si no es diferenciable.

Mantiene un intervalo [a,b] con DOS puntos interiores x1 < x2, colocados
con la razón áurea  phi = (sqrt(5)-1)/2 ~= 0.618  de forma que en cada
iteración solo hace falta UNA evaluación nueva de f (el otro punto
interior ya se había evaluado en la iteración anterior -- esa es la
propiedad clave que hace eficiente al método).

Regla: si f(x1) < f(x2), el mínimo está en [a, x2] (descartamos la
derecha); si f(x1) > f(x2), está en [x1, b] (descartamos la izquierda).

Función de ejemplo: la misma que en bisección, f(x) = x^2 - 2x + 3,
mínimo exacto en x*=1 -- así se puede comparar directamente el número
de iteraciones/evaluaciones que necesita cada método.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from _util import savepath

f = lambda x: x**2 - 2 * x + 3
PHI = (np.sqrt(5) - 1) / 2   # ~0.618


# ======================================================================
# 1) Implementación MANUAL
# ======================================================================
def busqueda_aurea(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    x1 = b - PHI * (b - a)
    x2 = a + PHI * (b - a)
    f1, f2 = f(x1), f(x2)
    historia = []

    for it in range(1, max_iter + 1):
        historia.append((it, a, b, x1, x2, f1, f2))
        if verbose and it <= 12:
            print(f"it={it:2d}  a={a: .6f}  b={b: .6f}  x1={x1: .6f} "
                  f"x2={x2: .6f}  f(x1)={f1: .6f}  f(x2)={f2: .6f}  "
                  f"ancho={b-a:.6f}")
        if (b - a) < tol:
            break
        if f1 < f2:
            # el minimo esta en [a, x2]; x1 pasa a ser el nuevo x2
            b = x2
            x2, f2 = x1, f1
            x1 = b - PHI * (b - a)
            f1 = f(x1)
        else:
            # el minimo esta en [x1, b]; x2 pasa a ser el nuevo x1
            a = x1
            x1, f1 = x2, f2
            x2 = a + PHI * (b - a)
            f2 = f(x2)

    xc = 0.5 * (a + b)
    return xc, f(xc), historia


if __name__ == "__main__":
    a0, b0 = -2.0, 5.0
    x_star, f_star, historia = busqueda_aurea(f, a0, b0, tol=1e-6)
    print(f"\nManual  -> x* = {x_star:.6f}   f(x*) = {f_star:.6f}  "
          f"({len(historia)} iteraciones, {len(historia)+1} evaluaciones de f)")

    # ======================================================================
    # 2) Comparación con SCIPY
    # ======================================================================
    res = minimize_scalar(f, bracket=(a0, (a0 + b0) / 2, b0), method="golden",
                           options={"xtol": 1e-6})
    print(f"scipy   -> x* = {res.x:.6f}   f(x*) = {res.fun:.6f}   "
          f"(scipy.optimize.minimize_scalar, method='golden')")

    # ======================================================================
    # 3) Figura: convergencia (para comparar luego con bisección)
    # ======================================================================
    import matplotlib.pyplot as plt
    anchos = [b - a for (_, a, b, _, _, _, _) in historia]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(range(1, len(anchos) + 1), anchos, "o-", color="#d62728",
                label="ancho del intervalo $b-a$")
    ax.set_xlabel("iteración")
    ax.set_ylabel("ancho del intervalo (escala log)")
    ax.set_title(r"Búsqueda áurea: el intervalo se reduce por un factor "
                 r"$\phi\approx0.618$ por iteración")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath("fig02_aurea_convergencia.png"), dpi=150)
    print("\nFigura:", savepath("fig02_aurea_convergencia.png"))
