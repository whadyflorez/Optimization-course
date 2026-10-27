"""
Función objetivo de 2 variables con 6 mínimos locales, la misma familia de
"pozos" usada al explicar basin-hopping (se reutiliza aquí para grid
search y random walk, y se retomará en la Clase 3 con las metaheurísticas
avanzadas -- así los estudiantes ven los mismos métodos "básicos" de la
Clase 2 aplicados al mismo paisaje difícil que motivará la Clase 3).
"""
import numpy as np

# cada pozo: (profundidad, centro_x, centro_y, ancho)
POZOS = [
    (-5.0, -6.5,  4.0, 1.3),   # <- el más profundo: mínimo GLOBAL
    (-4.0,  5.0,  4.0, 1.1),
    (-3.0, -5.0, -4.0, 1.1),
    (-2.0,  5.0, -5.0, 1.0),
    (-1.8,  0.0,  6.0, 1.0),
    (-1.5, -6.0,  0.0, 1.0),
]


def f(x, y):
    z = 0.03 * (x**2 + y**2)
    for w, cx, cy, s in POZOS:
        z += w * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * s**2))
    return z
