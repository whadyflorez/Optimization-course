"""
Método de Newton con Hessiana aproximada (métodos quasi-Newton): BFGS.

Motivación: Newton exacto requiere calcular y factorizar H(x_k) (costoso,
y a veces H ni siquiera está disponible analíticamente). Los métodos
quasi-Newton construyen una APROXIMACIÓN B_k ~ H(x_k) (o su inversa
directamente) usando solo información de gradiente ya calculada en pasos
anteriores.

Actualización BFGS (Broyden-Fletcher-Goldfarb-Shanno) de la inversa H_k^{-1}:
    s_k = x_{k+1} - x_k                  (cambio en la posición)
    y_k = grad(x_{k+1}) - grad(x_k)      (cambio en el gradiente)

    H_{k+1}^{-1} = (I - rho_k s_k y_k^T) H_k^{-1} (I - rho_k y_k s_k^T) + rho_k s_k s_k^T
    con rho_k = 1 / (y_k^T s_k)

Propiedades clave:
  - Se inicializa H_0^{-1} = I (equivalente a un paso de descenso de gradiente
    al inicio).
  - Cada paso: p_k = -H_k^{-1} grad(x_k), luego line search (Wolfe/Armijo) en
    esa dirección.
  - Si y_k^T s_k > 0 (condición de curvatura), la actualización preserva
    definición positiva de H_k^{-1} automáticamente -> siempre dirección de
    descenso, sin necesidad de verificar autovalores como en Newton puro.
  - Convergencia superlineal (más lento que Newton cuadrático, pero sin
    necesitar la Hessiana).
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import rosen_f, rosen_grad
from _plotting import plot_trajectory, plot_convergence


def armijo_line_search(f, grad, x, p, alpha0=1.0, c1=1e-4, rho=0.5):
    phi0 = f(x)
    dphi0 = grad(x) @ p
    alpha = alpha0
    while f(x + alpha * p) > phi0 + c1 * alpha * dphi0:
        alpha *= rho
        if alpha < 1e-16:
            break
    return alpha


def bfgs_method(f, grad, x0, tol=1e-10, max_iter=200):
    x = np.array(x0, dtype=float)
    n = len(x)
    Hinv = np.eye(n)  # aproximación inicial de la Hessiana inversa
    g = grad(x)
    traj = [x.copy()]
    fvals = [f(x)]
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        p = -Hinv @ g
        alpha = armijo_line_search(f, grad, x, p)
        x_new = x + alpha * p
        g_new = grad(x_new)

        s = x_new - x
        y = g_new - g
        sy = y @ s
        if sy > 1e-12:  # condición de curvatura: actualizar solo si se cumple
            rho_k = 1.0 / sy
            I = np.eye(n)
            V = I - rho_k * np.outer(s, y)
            Hinv = V @ Hinv @ V.T + rho_k * np.outer(s, s)
        # si sy <= 0, se conserva Hinv (salvaguarda de robustez)

        x, g = x_new, g_new
        traj.append(x.copy())
        fvals.append(f(x))
    return np.array(traj), fvals, k + 1


if __name__ == "__main__":
    x0 = [-1.5, 2.0]
    traj, fvals, n_iter = bfgs_method(rosen_f, rosen_grad, x0, max_iter=200)
    print(f"BFGS manual: {n_iter} iteraciones, x_final={traj[-1]}, f_final={fvals[-1]:.3e}")

    plot_trajectory(rosen_f, traj, xlim=(-2, 2), ylim=(-1, 3),
                     title="Quasi-Newton (BFGS)\nRosenbrock",
                     fname=figpath("fig06_bfgs_rosenbrock.png"),
                     levels=np.geomspace(0.5, 2000, 25), minimum=(1, 1))

    plot_convergence([fvals], ["BFGS (manual)"],
                      "Convergencia superlineal: BFGS", figpath("fig06_bfgs_convergencia.png"))

    res = minimize(rosen_f, x0, jac=rosen_grad, method="BFGS")
    print(f"scipy BFGS: x*={res.x}, f*={res.fun:.3e}, iters={res.nit}")
    print("\nComparaci\u00f3n de iteraciones -> Newton exacto: 24 | BFGS: "
          f"{n_iter} | GD+Armijo: 3000  (script 03/05)")
