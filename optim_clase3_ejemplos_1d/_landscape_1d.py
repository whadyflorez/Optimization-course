"""
Función objetivo de 1 variable con 5 mínimos locales, pensada como la
versión 1D del paisaje de 6 pozos usado en el resto de la Clase 3
(_landscape.py). Mismo espíritu: varios mínimos locales de profundidad
distinta, uno de ellos claramente el global, y un mínimo local "trampa"
cerca del punto de arranque para que el descenso puro se quede atascado.
"""
import numpy as np

# cada pozo: (profundidad, centro, ancho)
POZOS_1D = [
    (-5.0, -6.5, 1.3),   # <- el más profundo: mínimo GLOBAL, x* ~ -6.41
    (-4.0,  5.0, 1.0),   # trampa cercana al punto de arranque
    (-3.2,  1.0, 0.8),
    (-2.5, -1.5, 0.7),
    (-2.0,  8.0, 1.0),   # pozo somero, el más cercano al arranque x0=8.5
]


def f(x):
    x = np.asarray(x, dtype=float)
    z = 0.02 * x**2
    for w, c, s in POZOS_1D:
        z = z + w * np.exp(-((x - c) ** 2) / (2 * s**2))
    return z
