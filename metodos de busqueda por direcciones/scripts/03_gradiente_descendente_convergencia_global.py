"""
Descenso de gradiente con GARANTÍA de convergencia global: paso variable
elegido en cada iteración mediante backtracking Armijo (en vez de un alpha
fijo elegido a mano, que como vimos en 01 puede divergir).

Algoritmo:
  1. d_k = -grad(f(x_k))                      (dirección de descenso)
  2. alpha_k = backtracking Armijo desde alpha0 (line search inexacto)
  3. x_{k+1} = x_k + alpha_k * d_k
  4. repetir hasta ||grad|| < tol

"Convergencia global" aquí significa: converge a un punto estacionario desde
CUALQUIER punto inicial (no que encuentre el mínimo global de una función
no convexa) — es la terminología estándar de optimización numérica
(Nocedal & Wright, cap. 3).

Se prueba en Rosenbrock (no convexa, valle curvo) para mostrar que el paso fijo
sería casi imposible de elegir a mano en toda la trayectoria, mientras que
Armijo se adapta automáticamente.
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import rosen_f, rosen_grad
from _plotting import plot_trajectory, plot_convergence


def gradient_descent_armijo(f, grad, x0, alpha0=1.0, c1=1e-4, rho=0.5,
                             tol=1e-6, max_iter=5000):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    fvals = [f(x)]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        d = -g
        alpha = alpha0
        phi0 = f(x)
        dphi0 = g @ d
        while f(x + alpha * d) > phi0 + c1 * alpha * dphi0:
            alpha *= rho
            if alpha < 1e-16:
                break
        x = x + alpha * d
        traj.append(x.copy())
        fvals.append(f(x))
    return np.array(traj), fvals, k + 1


if __name__ == "__main__":
    x0 = [-1.5, 2.0]
    traj, fvals, n_iter = gradient_descent_armijo(rosen_f, rosen_grad, x0,
                                                    max_iter=3000, tol=1e-5)
    print(f"Descenso de gradiente + Armijo: {n_iter} iteraciones, "
          f"x_final={traj[-1]}, f_final={fvals[-1]:.3e}")

    plot_trajectory(rosen_f, traj, xlim=(-2, 2), ylim=(-1, 3),
                     title="Descenso de gradiente con paso variable (Armijo)\nRosenbrock",
                     fname=figpath("fig03_gd_armijo_rosenbrock.png"),
                     levels=np.geomspace(0.5, 2000, 25), minimum=(1, 1))

    plot_convergence([fvals], ["GD + Armijo"],
                      "Convergencia: descenso de gradiente con line search",
                      figpath("fig03_gd_armijo_convergencia.png"))

    # Comparación con scipy (CG, que también usa line search de Wolfe)
    res = minimize(rosen_f, x0, jac=rosen_grad, method="CG",
                    options={"maxiter": 3000})
    print(f"scipy CG: x*={res.x}, f*={res.fun:.3e}, iters={res.nit}")
    print(f"\nNota pedag\u00f3gica: GD+Armijo necesita ~{n_iter} iteraciones para "
          f"{res.nit} de CG en el mismo valle curvo -> motiva Newton/quasi-Newton.")
