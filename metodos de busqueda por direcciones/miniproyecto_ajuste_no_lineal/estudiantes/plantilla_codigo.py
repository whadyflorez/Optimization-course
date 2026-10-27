"""
PLANTILLA DEL MINIPROYECTO - Ajuste no lineal por minimos cuadrados
Modelo:  y(t) = a*exp(-b*t) + c*exp(-d*t)       theta = (a, b, c, d)

Completa las funciones marcadas con  TODO.  Uso:
    python plantilla_codigo.py datos_<documento>.csv
(sin argumento usa datos_ejemplo.csv).  Solo se permite numpy dentro de tus
implementaciones; scipy se usa unicamente para comparar.
"""
import sys
import numpy as np
from scipy.optimize import least_squares


# ------------------------------------------------------------ datos y modelo
def cargar(nombre):
    d = np.genfromtxt(nombre, delimiter=",", names=True)
    return d["t"], d["y"]


def modelo(th, t):
    a, b, c, d = th
    return a * np.exp(-b * t) + c * np.exp(-d * t)


def residuos(th, t, y):
    """r_i = modelo(t_i) - y_i"""
    return modelo(th, t) - y


def costo(th, t, y):
    r = residuos(th, t, y)
    return 0.5 * r @ r


def jacobiana(th, t):
    """TODO (Parte A1): matriz n x 4 con J[i, j] = d r_i / d theta_j."""
    raise NotImplementedError


def verificar_jacobiana(th, t, y, h=1e-7):
    """TODO (Parte A3): compara jacobiana() con diferencias finitas centrales.
    Devuelve el error relativo maximo."""
    raise NotImplementedError


def datos_sinteticos(th, sigma, n=150, t_max=20.0, seed=0):
    """Datos del modelo con ruido N(0, sigma^2) (para la Parte D3)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, t_max, n))
    t[0] = 0.0
    return t, modelo(th, t) + sigma * rng.standard_normal(n)


# ----------------------------------------------------------- minimizadores
def dogleg_ls(th0, t, y, Delta0=1.0, Delta_max=100.0, tol=1e-6, max_iter=200):
    """TODO (Parte B1). Debe devolver (historial_theta, historial_costo, nit)."""
    raise NotImplementedError


def lm_ls(th0, t, y, lam0=1e-2, tol=1e-6, max_iter=200):
    """TODO (Parte B2). Debe devolver (historial_theta, historial_costo, nit)."""
    raise NotImplementedError


def sgd_ls(th0, t, y, batch=16, lr0=0.02, k0=2000, epocas=300, seed=0, beta=0.0):
    """TODO (Parte B3). Debe devolver (historial_theta, historial_costo, pasos);
    guarda una entrada del historial por EPOCA."""
    raise NotImplementedError


# --------------------------------------------------------------------- main
if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else "datos_ejemplo.csv"
    t, y = cargar(archivo)
    th0 = np.array([1.0, 1.0, 1.0, 0.5])

    # --- referencia (ya implementada): usar para medir tus resultados
    ref = least_squares(lambda th: residuos(th, t, y), th0, method="lm",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    th_star, f_star = ref.x, 0.5 * ref.fun @ ref.fun
    print(f"Referencia scipy: theta*={np.round(th_star, 5)}  f*={f_star:.6f}")

    for nombre, fun in (("Dogleg", dogleg_ls), ("LM", lm_ls)):
        try:
            H, C, nit = fun(th0, t, y)
            print(f"{nombre}: {nit} iteraciones, f={C[-1]:.6f}, "
                  f"|theta-theta*|={np.linalg.norm(H[-1] - th_star):.2e}")
        except NotImplementedError:
            print(f"{nombre}: pendiente (TODO)")
    try:
        H, C, k = sgd_ls(th0, t, y)
        print(f"SGD: {k} pasos, (f-f*)/f*={(C[-1] - f_star) / f_star:.3f}")
    except NotImplementedError:
        print("SGD: pendiente (TODO)")
