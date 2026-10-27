"""
Método de Newton para optimización (usa gradiente y Hessiana exactos).

Modelo cuadrático local (Taylor de 2do orden) en x_k:
    m_k(p) = f(x_k) + grad(x_k)^T p + 0.5 * p^T H(x_k) p

Minimizando m_k respecto a p (asumiendo H(x_k) definida positiva):
    grad_p m_k(p) = grad(x_k) + H(x_k) p = 0   =>   p_k = -H(x_k)^{-1} grad(x_k)

    x_{k+1} = x_k + p_k          (paso de Newton puro, alpha=1)

Con función cuadrática exacta, un solo paso basta (converge en 1 iteración).
Con Rosenbrock (no cuadrática), Newton converge muy rápido cerca del mínimo
(orden cuadrático) pero puede fallar lejos del mínimo si H no es definida
positiva (dirección p_k podría no ser de descenso) — se ilustra con line
search de respaldo (Armijo) para dar robustez ("Newton amortiguado").
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import rosen_f, rosen_grad, rosen_hess, quad_f, quad_grad, quad_hess
from _plotting import plot_trajectory, plot_convergence


def newton_method(f, grad, hess, x0, tol=1e-10, max_iter=50, damped=True):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    fvals = [f(x)]
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = -g  # H singular: retroceder a descenso de gradiente

        if damped:
            # Newton "amortiguado": si p no es de descenso, usar -g;
            # luego backtracking Armijo sobre el paso alpha (protege
            # contra Hessianas indefinidas lejos del mínimo).
            if g @ p >= 0:
                p = -g
            alpha = 1.0
            phi0 = f(x)
            dphi0 = g @ p
            while f(x + alpha * p) > phi0 + 1e-4 * alpha * dphi0 and alpha > 1e-12:
                alpha *= 0.5
        else:
            alpha = 1.0

        x = x + alpha * p
        traj.append(x.copy())
        fvals.append(f(x))
    return np.array(traj), fvals, k + 1


if __name__ == "__main__":
    print("--- Newton puro en cuadr\u00e1tica (converge en 1 paso) ---")
    traj_q, fvals_q, n_q = newton_method(quad_f, quad_grad, quad_hess, [4.0, 0.9],
                                          damped=False)
    print(f"Iteraciones: {n_q}, x_final={traj_q[-1]}, f_final={fvals_q[-1]:.3e}")
    plot_trajectory(quad_f, traj_q, xlim=(-5, 5), ylim=(-3, 3),
                     title="Newton puro en cuadr\u00e1tica\n(1 paso, exacto)",
                     fname=figpath("fig05_newton_cuadratica.png"), minimum=(0, 0))

    print("\n--- Newton amortiguado en Rosenbrock ---")
    x0 = [-1.5, 2.0]
    traj_r, fvals_r, n_r = newton_method(rosen_f, rosen_grad, rosen_hess, x0,
                                          damped=True, max_iter=100)
    print(f"Iteraciones: {n_r}, x_final={traj_r[-1]}, f_final={fvals_r[-1]:.3e}")
    plot_trajectory(rosen_f, traj_r, xlim=(-2, 2), ylim=(-1, 3),
                     title="M\u00e9todo de Newton amortiguado\nRosenbrock",
                     fname=figpath("fig05_newton_rosenbrock.png"),
                     levels=np.geomspace(0.5, 2000, 25), minimum=(1, 1))

    plot_convergence([fvals_r], ["Newton amortiguado"],
                      "Convergencia cuadr\u00e1tica cerca del m\u00ednimo (Newton)",
                      figpath("fig05_newton_convergencia.png"))

    res = minimize(rosen_f, x0, jac=rosen_grad, hess=rosen_hess, method="Newton-CG")
    print(f"scipy Newton-CG: x*={res.x}, f*={res.fun:.3e}, iters={res.nit}")
