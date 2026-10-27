"""
Método NELDER-MEAD ("amoeba") en 2D: optimización derivative-free que
mantiene un SIMPLEX (en 2D: un triángulo, 3 puntos) y lo va deformando
-reflejando, expandiendo, contrayendo o encogiendo- para "arrastrarlo"
hacia el mínimo, como una ameba que se mueve cambiando de forma.

En cada iteración, con los 3 vértices ordenados por f (mejor B, medio M,
peor P):
  1. Centroide C de los puntos que NO son el peor (aquí: B y M).
  2. Reflexión: R = C + alpha*(C - P)  (alpha=1 típicamente).
     - Si f(R) es mejor que el mejor (f(R) < f(B)): probar EXPANSIÓN
       E = C + gamma*(R - C) (gamma=2); quedarse con el mejor de R,E.
     - Si f(R) es mejor que el segundo peor pero no el mejor: aceptar R.
     - Si f(R) es peor que el segundo peor: CONTRACCIÓN hacia adentro,
       Ct = C + rho*(P - C) (rho=0.5); si mejora, aceptar Ct.
     - Si nada mejora: ENCOGER todo el simplex hacia el mejor punto B.

Función de ejemplo: la misma de la Clase 1, f(x,y)=  (mínimo
único y convexo en el origen) -- así se ve limpio cómo el triángulo se
va encogiendo y desplazando hacia (0,0).
"""
import numpy as np
from scipy.optimize import minimize
from _util import savepath

f = lambda p: (p[0]-2)**2 + (p[1]+1)**2 + 5*np.sin(p[0]) + 3*np.cos(p[1])


# ======================================================================
# 1) Implementación MANUAL
# ======================================================================
def nelder_mead(f, x0, paso_inicial=1.0, tol=1e-8, max_iter=100, verbose=True):
    n = len(x0)
    # simplex inicial: x0 y n puntos desplazados un paso en cada eje
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += paso_inicial
        simplex.append(p)

    historia = [np.array(simplex)]

    for it in range(1, max_iter + 1):
        simplex.sort(key=f)                     # ordena: mejor -> peor
        B, *resto, P = simplex                    # B=mejor, P=peor
        fB, fP = f(B), f(P)

        if abs(fP - fB) < tol:
            break

        C = np.mean(simplex[:-1], axis=0)          # centroide sin el peor
        R = C + 1.0 * (C - P)                        # reflexion
        fR = f(R)

        if verbose and it <= 10:
            print(f"it={it:2d}  mejor={B.round(3)} f={fB:.5f}   "
                  f"peor={P.round(3)} f={fP:.5f}   ", end="")

        if fR < fB:
            E = C + 2.0 * (R - C)                    # expansion
            if f(E) < fR:
                simplex[-1] = E
                accion = "EXPANSION"
            else:
                simplex[-1] = R
                accion = "REFLEXION (aceptada por expansion)"
        elif fR < f(resto[-1]):
            simplex[-1] = R
            accion = "REFLEXION"
        else:
            Ct = C + 0.5 * (P - C)                   # contraccion
            if f(Ct) < fP:
                simplex[-1] = Ct
                accion = "CONTRACCION"
            else:
                simplex = [B + 0.5 * (v - B) for v in simplex]  # encoger
                accion = "ENCOGER (shrink)"

        if verbose and it <= 10:
            print(accion)

        historia.append(np.array(simplex))

    simplex.sort(key=f)
    return simplex[0], f(simplex[0]), historia


if __name__ == "__main__":
    x0 = [4.0, 3.0]
    x_star, f_star, historia = nelder_mead(f, x0, paso_inicial=1.5, tol=1e-8)
    print(f"\nManual  -> x* = {x_star.round(6)}   f(x*) = {f_star:.2e}  "
          f"({len(historia)-1} iteraciones)")

    # ======================================================================
    # 2) Comparación con SCIPY
    # ======================================================================
    res = minimize(f, x0, method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-8})
    print(f"scipy   -> x* = {res.x.round(6)}   f(x*) = {res.fun:.2e}   "
          f"({res.nit} iteraciones, scipy.optimize.minimize method='Nelder-Mead')")

    # ======================================================================
    # 3) Figura: evolución del simplex sobre las curvas de nivel
    # ======================================================================
    import matplotlib.pyplot as plt
    xs = np.linspace(-1, 5, 300)
    ys = np.linspace(-1, 5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = (X-2)**2 + (Y+1)**2 + 5*np.sin(X) + 3*np.cos(Y)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, Z, levels=15, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    pasos_a_mostrar = sorted(set(list(range(0, min(8, len(historia)))) +
                                  [len(historia) - 1]))
    cmap = plt.cm.plasma(np.linspace(0, 1, len(pasos_a_mostrar)))
    for color, k in zip(cmap, pasos_a_mostrar):
        tri = historia[k]
        pts = np.vstack([tri, tri[0]])   # cerrar el triangulo
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, ms=4, lw=1.5,
                 label=f"it {k}")

    ax.plot(*x0, "s", color="cyan", ms=10, mec="black", zorder=5)
    ax.plot(x_star[0],x_star[1], "*", color="gold", ms=20, mec="black", zorder=5,
            label="mínimo real (0,0)")
    ax.set_title("Nelder-Mead (amoeba): el simplex se encoge hacia el mínimo")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(savepath("fig06_nelder_mead.png"), dpi=150)
    print("\nFigura:", savepath("fig06_nelder_mead.png"))
