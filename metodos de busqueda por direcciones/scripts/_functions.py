"""
Funciones de prueba compartidas por todos los scripts de la Clase 4
(métodos basados en dirección de búsqueda).

1) Cuadrática mal condicionada:  f(x,y) = 0.5*(x^2 + kappa*y^2)
   - Convexa, mínimo único en el origen.
   - kappa controla el número de condición de la Hessiana (kappa=1 -> círculos,
     kappa grande -> elipses muy alargadas -> zig-zag de descenso de gradiente).

2) Rosenbrock (función "banana"):  f(x,y) = (a-x)^2 + b*(y-x^2)^2
   - No convexa, valle curvo muy estrecho, mínimo global en (a, a^2).
   - Estándar en la literatura (Newton, quasi-Newton, trust region, Powell).
   - Se usa también en forma de mínimos cuadrados r(x)=[a-x, sqrt(b)*(y-x^2)]
     para Levenberg-Marquardt.
"""
import numpy as np

# ---------- Cuadrática mal condicionada ----------
KAPPA = 25.0


def quad_f(xy):
    x, y = xy
    return 0.5 * (x**2 + KAPPA * y**2)


def quad_grad(xy):
    x, y = xy
    return np.array([x, KAPPA * y])


def quad_hess(xy):
    return np.array([[1.0, 0.0], [0.0, KAPPA]])


# ---------- Rosenbrock ----------
ROSEN_A = 1.0
ROSEN_B = 100.0


def rosen_f(xy):
    x, y = xy
    return (ROSEN_A - x) ** 2 + ROSEN_B * (y - x**2) ** 2


def rosen_grad(xy):
    x, y = xy
    dfdx = -2 * (ROSEN_A - x) - 4 * ROSEN_B * x * (y - x**2)
    dfdy = 2 * ROSEN_B * (y - x**2)
    return np.array([dfdx, dfdy])


def rosen_hess(xy):
    x, y = xy
    d2fdx2 = 2 - 4 * ROSEN_B * (y - x**2) + 8 * ROSEN_B * x**2
    d2fdxdy = -4 * ROSEN_B * x
    d2fdy2 = 2 * ROSEN_B
    return np.array([[d2fdx2, d2fdxdy], [d2fdxdy, d2fdy2]])


def rosen_residuals(xy):
    """Forma de mínimos cuadrados: f = ||r(x)||^2, r en R^2."""
    x, y = xy
    return np.array([ROSEN_A - x, np.sqrt(ROSEN_B) * (y - x**2)])


def rosen_jacobian(xy):
    """Jacobiano de r(x) respecto a (x,y), 2x2."""
    x, y = xy
    return np.array([[-1.0, 0.0], [-2 * np.sqrt(ROSEN_B) * x, np.sqrt(ROSEN_B)]])
