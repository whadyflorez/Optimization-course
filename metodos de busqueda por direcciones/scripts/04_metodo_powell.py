"""
Método de Powell (1964): minimización direccional sucesiva sin usar gradiente,
construyendo direcciones conjugadas.

Idea del ciclo (para n=2 variables, direcciones iniciales e1=(1,0), e2=(0,1)):
  1. Desde x0, line search exacto a lo largo de d1=e1  -> x1
  2. Desde x1, line search exacto a lo largo de d2=e2  -> x2
  3. Nueva dirección conjugada: d_new = x2 - x0
  4. Line search a lo largo de d_new desde x2          -> x3
  5. Se descarta la dirección que más contribuyó a la reducción de f
     (heurística de Powell) y se reemplaza por d_new; se repite el ciclo
     con el nuevo conjunto de direcciones {d2, d_new} (en 2D, simplemente
     rota el conjunto).

Cada "line search a lo largo de una dirección" es exactamente el problema
1D de line search visto en el script 02 (aquí resuelto con
scipy.optimize.minimize_scalar, ya que Powell por definición no requiere
derivadas analíticas).
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import rosen_f
from _plotting import plot_trajectory, plot_convergence


def line_search_along(f, x, d, bracket=(-2, 2)):
    """Line search 1D exacto a lo largo de d, con salvaguarda: si el resultado
    de Brent es peor que quedarse en alpha=0 (puede pasar en valles muy
    curvos, donde phi(alpha) no es unimodal en el bracket dado), se
    conserva el punto de partida."""
    def phi(alpha):
        return f(x + alpha * d)
    res = minimize_scalar(phi, bracket=bracket)
    if res.fun <= phi(0.0):
        return x + res.x * d, res.fun
    return x.copy(), phi(0.0)


def powell_method(f, x0, n_iter=25, tol=1e-8):
    x = np.array(x0, dtype=float)
    n = len(x)
    directions = list(np.eye(n))
    traj = [x.copy()]
    fvals = [f(x)]
    for it in range(n_iter):
        x_start = x.copy()
        f_start = f(x)
        reductions = []
        for d in directions:
            x_new, f_new = line_search_along(f, x, d)
            reductions.append(f(x) - f_new)  # reducción lograda por esta dirección
            x = x_new
            traj.append(x.copy())
            fvals.append(f(x))
        # nueva dirección conjugada
        d_new = x - x_start
        norm_d = np.linalg.norm(d_new)
        if norm_d < 1e-14:
            break
        d_new = d_new / norm_d
        # extrapolación + line search a lo largo de d_new
        x_new, f_new = line_search_along(f, x, d_new)
        # regla de Powell: reemplazar la dirección que más redujo f
        idx_max = int(np.argmax(reductions))
        directions[idx_max] = d_new
        x = x_new
        traj.append(x.copy())
        fvals.append(f(x))
        if abs(f_start - f(x)) < tol:
            break
    return np.array(traj), fvals, it + 1


if __name__ == "__main__":
    x0 = [-1.5, 2.0]
    traj, fvals, n_iter = powell_method(rosen_f, x0, n_iter=30)
    print(f"Powell manual: {n_iter} ciclos, {len(traj)-1} evaluaciones de punto, "
          f"x_final={traj[-1]}, f_final={fvals[-1]:.3e}")

    plot_trajectory(rosen_f, traj, xlim=(-2, 2), ylim=(-1, 3),
                     title="M\u00e9todo de Powell (direcciones conjugadas)\nRosenbrock",
                     fname=figpath("fig04_powell_rosenbrock.png"),
                     levels=np.geomspace(0.5, 2000, 25), minimum=(1, 1),
                     extra_label="Puntos de b\u00fasqueda 1D")

    plot_convergence([fvals], ["Powell (manual)"],
                      "Convergencia: m\u00e9todo de Powell", figpath("fig04_powell_convergencia.png"))

    res = minimize(rosen_f, x0, method="Powell")
    print(f"scipy Powell: x*={res.x}, f*={res.fun:.3e}, "
          f"iters={res.nit}, evals={res.nfev}")
