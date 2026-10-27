"""
Ejemplo principal (Wyndor Glass Co., Hillier & Lieberman):

    max  z = 3 x1 + 5 x2
    s.a. x1            <= 4
              2 x2      <= 12
         3 x1 + 2 x2    <= 18
         x1, x2 >= 0

Verifica a mano (con fracciones exactas via Fraction) las dos iteraciones
del simplex que se muestran en las diapositivas, y compara el óptimo
contra scipy.optimize.linprog.
"""
from fractions import Fraction as F
from scipy.optimize import linprog
import numpy as np


def print_row(label, row, rhs):
    terms = " + ".join(f"{row[j]}*{name}" for j, name in enumerate(cols) if row[j] != 0)
    print(f"  {label}: {terms} = {rhs}")


# Variables (orden): x1, x2, s1, s2, s3
cols = ["x1", "x2", "s1", "s2", "s3"]

# Tableau inicial (filas de restricciones + fila z), todo en Fraction
# fila z se guarda como coeficientes de (x1,x2,s1,s2,s3) tales que z - c^T x = 0
rows = {
    "s1": ([F(1), F(0), F(1), F(0), F(0)], F(4)),
    "s2": ([F(0), F(2), F(0), F(1), F(0)], F(12)),
    "s3": ([F(3), F(2), F(0), F(0), F(1)], F(18)),
}
z_row = ([F(-3), F(-5), F(0), F(0), F(0)], F(0))  # z - 3x1 - 5x2 = 0
basis = ["s1", "s2", "s3"]

print("=== Tableau inicial (base = s1,s2,s3) ===")
for name in basis:
    print_row(name, *rows[name])
print_row("z ", *z_row)
print()

def pivot(rows, z_row, basis, entering_idx, leaving_name):
    piv_row_name = leaving_name
    piv_coeffs, piv_rhs = rows[piv_row_name]
    piv_val = piv_coeffs[entering_idx]
    new_piv = [c / piv_val for c in piv_coeffs]
    new_piv_rhs = piv_rhs / piv_val
    rows = dict(rows)
    rows[piv_row_name] = (new_piv, new_piv_rhs)
    for name in list(rows.keys()):
        if name == piv_row_name:
            continue
        coeffs, rhs = rows[name]
        factor = coeffs[entering_idx]
        if factor != 0:
            new_coeffs = [c - factor * npc for c, npc in zip(coeffs, new_piv)]
            new_rhs = rhs - factor * new_piv_rhs
            rows[name] = (new_coeffs, new_rhs)
    zc, zr = z_row
    factor = zc[entering_idx]
    if factor != 0:
        zc = [c - factor * npc for c, npc in zip(zc, new_piv)]
        zr = zr - factor * new_piv_rhs
    z_row = (zc, zr)
    basis = [cols[entering_idx] if b == piv_row_name else b for b in basis]
    # rename key in dict
    rows[cols[entering_idx]] = rows.pop(piv_row_name)
    return rows, z_row, basis


# --- Iteracion 1: entra x2 (coef mas negativo en z: -5), min ratio test ---
entering = cols.index("x2")
ratios = {}
for name in basis:
    coeffs, rhs = rows[name]
    if coeffs[entering] > 0:
        ratios[name] = rhs / coeffs[entering]
print("Iteracion 1: entra x2. Razones:", {k: str(v) for k, v in ratios.items()})
leaving = min(ratios, key=ratios.get)
print("Sale:", leaving)
rows, z_row, basis = pivot(rows, z_row, basis, entering, leaving)
print("\n=== Tableau tras iteracion 1 (base =", basis, ") ===")
for name in basis:
    print_row(name, *rows[name])
print_row("z ", *z_row)
print()

# --- Iteracion 2: entra x1 ---
entering = cols.index("x1")
ratios = {}
for name in basis:
    coeffs, rhs = rows[name]
    if coeffs[entering] > 0:
        ratios[name] = rhs / coeffs[entering]
print("Iteracion 2: entra x1. Razones:", {k: str(v) for k, v in ratios.items()})
leaving = min(ratios, key=ratios.get)
print("Sale:", leaving)
rows, z_row, basis = pivot(rows, z_row, basis, entering, leaving)
print("\n=== Tableau tras iteracion 2 (base =", basis, ") -- OPTIMO ===")
for name in basis:
    print_row(name, *rows[name])
print_row("z ", *z_row)

solution = {name: F(0) for name in cols}
for name in basis:
    _, rhs = rows[name]
    solution[name] = rhs
zc, zr = z_row
print("\nSolucion optima (manual, exacta):", {k: str(v) for k, v in solution.items()})
print("z* =", zr)

# --- Verificacion con scipy ---
res = linprog(
    c=[-3, -5, 0, 0, 0],
    A_eq=[[1, 0, 1, 0, 0], [0, 2, 0, 1, 0], [3, 2, 0, 0, 1]],
    b_eq=[4, 12, 18],
    bounds=[(0, None)] * 5,
    method="highs",
)
print("\n=== scipy.optimize.linprog (highs) ===")
print("x* =", res.x, " z* =", -res.fun)

assert abs(float(zr) - (-res.fun)) < 1e-9
assert abs(float(solution["x1"]) - res.x[0]) < 1e-9
assert abs(float(solution["x2"]) - res.x[1]) < 1e-9
print("\nOK: coincide con scipy.")
