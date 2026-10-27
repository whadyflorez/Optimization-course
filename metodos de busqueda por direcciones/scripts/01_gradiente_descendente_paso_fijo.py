"""
Descenso de gradiente (steepest descent) con paso fijo alpha.

x_{k+1} = x_k - alpha * grad(f(x_k))

Se usa la cuadrática mal condicionada f(x,y) = 0.5*(x^2 + kappa*y^2), kappa=25,
para que el zig-zag característico del descenso de gradiente en valles alargados
sea visualmente claro. Se comparan 3 valores de alpha:
  - alpha chico: converge, pero muy lento.
  - alpha "bueno" (cerca del óptimo teórico 2/(L+mu)): converge razonablemente.
  - alpha grande: diverge (rebota cada vez más lejos).

Parte manual + comparación con scipy.optimize.minimize(method='CG', pero con
line search real) para contrastar el paso fijo contra un método con paso adaptado.
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import quad_f, quad_grad, KAPPA
from _plotting import plot_trajectory, plot_convergence


def gradient_descent_fixed(f, grad, x0, alpha, n_iter=40):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    fvals = [f(x)]
    for _ in range(n_iter):
        x = x - alpha * grad(x)
        traj.append(x.copy())
        fvals.append(f(x))
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e6:
            break
    return np.array(traj), fvals


if __name__ == "__main__":
    x0 = [4.0, 0.9]

    # Paso teórico óptimo para cuadrática con autovalores {1, kappa}: alpha* = 2/(1+kappa)
    alpha_opt = 2.0 / (1.0 + KAPPA)
    alphas = {
        "alpha peque\u00f1o (0.01)": 0.01,
        f"alpha \u00f3ptimo ({alpha_opt:.3f})": alpha_opt,
        "alpha grande (0.09, diverge)": 0.09,
    }

    trajs, fvals_all, labels = [], [], []
    for label, a in alphas.items():
        traj, fvals = gradient_descent_fixed(quad_f, quad_grad, x0, a, n_iter=60)
        trajs.append(traj)
        fvals_all.append(fvals)
        labels.append(label)
        print(f"{label}: iter={len(traj)-1}, f_final={fvals[-1]:.4e}")
        plot_trajectory(quad_f, traj, xlim=(-5, 5), ylim=(-3, 3),
                         title=f"Descenso de gradiente, paso fijo\n{label}",
                         fname=figpath(f"fig01_gd_fijo_{a:.4f}.png"),
                         minimum=(0, 0))

    # Convergencia (solo los dos que no divergen, para escala log legible)
    plot_convergence(fvals_all[:2], labels[:2],
                      "Convergencia: efecto del tama\u00f1o de paso",
                      figpath("fig01_gd_fijo_convergencia.png"))

    # Comparación con scipy: gradiente conjugado (usa line search internamente)
    res = minimize(quad_f, x0, jac=quad_grad, method="CG")
    print("\nscipy CG (con line search interno):")
    print(f"  x* = {res.x}, f* = {res.fun:.6e}, iters = {res.nit}")
