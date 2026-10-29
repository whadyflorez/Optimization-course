"""
Verifica los 3 ejemplos de la clase con scipy.optimize.milp (el solver de
programacion entera mixta de scipy, disponible desde la version 1.9).
Por dentro usa HiGHS con branch-and-bound + planos de corte -- el mismo
concepto que vimos a mano, pero optimizado en C++.
"""
from scipy.optimize import milp, LinearConstraint, Bounds

print("=" * 70)
print("EJEMPLO 1: max 5x1+4x2 s.a. 6x1+4x2<=24, x1+2x2<=6, x1,x2 enteros")
print("=" * 70)
res1 = milp(
    c=[-5, -4],                                   # milp SIEMPRE minimiza
    constraints=LinearConstraint([[6, 4], [1, 2]], -float("inf"), [24, 6]),
    integrality=[1, 1],                           # 1 = variable entera, para AMBAS
    bounds=Bounds(0, float("inf")),
)
print(f"x* = {res1.x}  z* = {-res1.fun}")

print("\n" + "=" * 70)
print("EJEMPLO 2: max 5x1+3x2 s.a. 2x1+3x2<=12, 3x1+x2<=9, x1,x2 enteros")
print("=" * 70)
res2 = milp(
    c=[-5, -3],
    constraints=LinearConstraint([[2, 3], [3, 1]], -float("inf"), [12, 9]),
    integrality=[1, 1],
    bounds=Bounds(0, float("inf")),
)
print(f"x* = {res2.x}  z* = {-res2.fun}")

print("\n" + "=" * 70)
print("EJEMPLO 3 (MIXTO): max 8x1+5x2 s.a. x1+x2<=6, 9x1+5x2<=45,")
print("                    x1 ENTERA, x2 CONTINUA")
print("=" * 70)
res3 = milp(
    c=[-8, -5],
    constraints=LinearConstraint([[1, 1], [9, 5]], -float("inf"), [6, 45]),
    integrality=[1, 0],                           # 1 = entera (x1), 0 = continua (x2)
    bounds=Bounds(0, float("inf")),
)
print(f"x* = {res3.x}  z* = {-res3.fun}")

# --- Verificacion cruzada contra lo obtenido a mano con branch & bound ---
import numpy as np
assert np.allclose(res1.x, [4, 0]) and abs(-res1.fun - 20) < 1e-6
assert np.allclose(res2.x, [2, 2]) and abs(-res2.fun - 16) < 1e-6
assert np.allclose(res3.x, [4, 1.8]) and abs(-res3.fun - 41) < 1e-6
print("\nOK: los 3 resultados coinciden exactamente con el branch & bound hecho a mano.")
