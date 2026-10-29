"""
Ejemplo degenerado:

    max  z = 2 x1 + 3 x2
    s.a. x1        <= 4   (s1)
              x2   <= 4   (s2)
         x1 +  x2  <= 4   (s3)
         x1, x2 >= 0

Al entrar x2, hay EMPATE en la prueba de la razon minima entre s2 y s3
(ambas dan razon 4). Regla de Bland: entre las empatadas, sale la de
MENOR INDICE. Con orden (x1,x2,s1,s2,s3) -> s2 (indice 3) tiene menor
indice que s3 (indice 4), así que sale s2.
Tras el pivote, la solucion basica factible resultante tiene una
variable basica en 0 (s3=0): es DEGENERADA.
"""
from fractions import Fraction as F
from scipy.optimize import linprog

cols = ["x1", "x2", "s1", "s2", "s3"]


def print_row(label, row, rhs):
    terms = " + ".join(f"{row[j]}*{name}" for j, name in enumerate(cols) if row[j] != 0)
    print(f"  {label}: {terms} = {rhs}")


rows = {
    "s1": ([F(1), F(0), F(1), F(0), F(0)], F(4)),
    "s2": ([F(0), F(1), F(0), F(1), F(0)], F(4)),
    "s3": ([F(1), F(1), F(0), F(0), F(1)], F(4)),
}
z_row = ([F(-2), F(-3), F(0), F(0), F(0)], F(0))
basis = ["s1", "s2", "s3"]

print("=== Tableau inicial ===")
for n in basis:
    print_row(n, *rows[n])
print_row("z ", *z_row)

entering = cols.index("x2")
ratios = {}
for name in basis:
    coeffs, rhs = rows[name]
    if coeffs[entering] > 0:
        ratios[name] = rhs / coeffs[entering]
print("\nEntra x2. Razones:", {k: str(v) for k, v in ratios.items()})
print("-> EMPATE entre s2 y s3 (ambas 4). Regla de Bland: sale la de menor indice => s2.")

leaving = "s2"


def pivot(rows, z_row, basis, entering_idx, leaving_name):
    piv_coeffs, piv_rhs = rows[leaving_name]
    piv_val = piv_coeffs[entering_idx]
    new_piv = [c / piv_val for c in piv_coeffs]
    new_piv_rhs = piv_rhs / piv_val
    rows = dict(rows)
    rows[leaving_name] = (new_piv, new_piv_rhs)
    for name in list(rows.keys()):
        if name == leaving_name:
            continue
        coeffs, rhs = rows[name]
        factor = coeffs[entering_idx]
        if factor != 0:
            rows[name] = (
                [c - factor * npc for c, npc in zip(coeffs, new_piv)],
                rhs - factor * new_piv_rhs,
            )
    zc, zr = z_row
    factor = zc[entering_idx]
    if factor != 0:
        zc = [c - factor * npc for c, npc in zip(zc, new_piv)]
        zr = zr - factor * new_piv_rhs
    z_row = (zc, zr)
    basis = [cols[entering_idx] if b == leaving_name else b for b in basis]
    rows[cols[entering_idx]] = rows.pop(leaving_name)
    return rows, z_row, basis


rows, z_row, basis = pivot(rows, z_row, basis, entering, leaving)
print(f"\n=== Tableau tras el pivote (base = {basis}) ===")
for n in basis:
    print_row(n, *rows[n])
print_row("z ", *z_row)
sol = {c: F(0) for c in cols}
for n in basis:
    _, rhs = rows[n]
    sol[n] = rhs
print("\nSolucion basica factible resultante:", {k: str(v) for k, v in sol.items()})
print("-> s3 =", sol["s3"], " (variable basica en 0: SOLUCION DEGENERADA)")

# Siguiente iteracion: revisar z-row para ver si sigue habiendo mejora
zc, zr = z_row
print("\nFila z tras el pivote:", [str(c) for c in zc], " z =", zr)
neg = [j for j, c in enumerate(zc) if c < 0]
if not neg:
    print("No quedan coeficientes negativos en z: ya es OPTIMA (a pesar de ser degenerada).")

print("\nz sigue teniendo coeficiente negativo en x1 (-2): NO es optima todavia.")
print("=== Iteracion siguiente: entra x1 ===")
entering = cols.index("x1")
ratios = {}
for name in basis:
    coeffs, rhs = rows[name]
    if coeffs[entering] > 0:
        ratios[name] = rhs / coeffs[entering]
print("Razones:", {k: str(v) for k, v in ratios.items()})
leaving = min(ratios, key=ratios.get)
print("Sale:", leaving, " (razon 0 => PIVOTE DEGENERADO: la base cambia pero z NO mejora en este paso)")
rows, z_row, basis = pivot(rows, z_row, basis, entering, leaving)
print(f"\n=== Tableau tras el pivote degenerado (base = {basis}) ===")
for n in basis:
    print_row(n, *rows[n])
print_row("z ", *z_row)
sol = {c: F(0) for c in cols}
for n in basis:
    _, rhs = rows[n]
    sol[n] = rhs
print("Solucion:", {k: str(v) for k, v in sol.items()})
zc, zr = z_row
neg = [j for j, c in enumerate(zc) if c < 0]
print("z =", zr, "-> OPTIMA" if not neg else "-> sigue sin ser optima")

res = linprog(
    c=[-2, -3, 0, 0, 0],
    A_eq=[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]],
    b_eq=[4, 4, 4],
    bounds=[(0, None)] * 5,
    method="highs",
)
print("\n=== scipy (highs) ===  x* =", res.x, " z* =", -res.fun)
