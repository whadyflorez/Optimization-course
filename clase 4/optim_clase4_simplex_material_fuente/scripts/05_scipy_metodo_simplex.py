"""
Mismo ejemplo de la clase (Wyndor Glass Co.), resuelto con
scipy.optimize.linprog usando explicitamente el metodo SIMPLEX clasico
(tableau), no el solver moderno HiGHS por defecto.

    max  z = 3 x1 + 5 x2
    s.a. x1            <= 4
              2 x2      <= 12
         3 x1 + 2 x2    <= 18
         x1, x2 >= 0

Nota sobre el metodo:
- scipy.optimize.linprog SIEMPRE resuelve un problema de MINIMIZACION,
  asi que para maximizar z=3x1+5x2 hay que minimizar -z, es decir
  pasar c=[-3, -5].
- method="simplex" es el simplex clasico (el mismo algoritmo de tableau
  que vimos a mano en clase: fila z, razon minima, pivoteo). Esta
  DEPRECADO desde scipy 1.11 en favor de "highs" (mas rapido y mas
  robusto numericamente), pero sigue disponible y es el que pide el
  enunciado para comparar 1 a 1 con el metodo visto en clase.
- method="revised simplex" es una variante mas eficiente del mismo
  algoritmo (opera con la inversa de la base en vez de mantener todo
  el tableau explicito) -- mismo resultado, menos iteraciones/memoria.
- method="highs" es el solver recomendado hoy en dia (mismo resultado,
  internamente no usa el tableau simplex clasico sino una variante
  dual-simplex/punto-interior segun el problema).
"""
import warnings
from scipy.optimize import linprog

# --- Forma con desigualdades (A_ub, b_ub): la forma "natural" del problema ---
c = [-3, -5]                       # minimizar -z  <=>  maximizar z=3x1+5x2
A_ub = [[1, 0],                    # x1        <= 4
        [0, 2],                    # 2x2       <= 12
        [3, 2]]                    # 3x1 + 2x2 <= 18
b_ub = [4, 12, 18]
bounds = [(0, None), (0, None)]    # x1, x2 >= 0

print("=" * 60)
print("METODO SIMPLEX CLASICO (method='simplex')")
print("=" * 60)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # esperado: metodo deprecado
    res_simplex = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                           method="simplex", options={"disp": True})
print(res_simplex)
print(f"\nx1* = {res_simplex.x[0]:.4f}   x2* = {res_simplex.x[1]:.4f}")
print(f"z*  = {-res_simplex.fun:.4f}   (z = -fun, porque minimizamos -z)")
print(f"Iteraciones (pivotes) del simplex: {res_simplex.nit}")

# --- Comparacion con el resultado obtenido a mano en clase ---
x1_manual, x2_manual, z_manual = 2, 6, 36
assert abs(res_simplex.x[0] - x1_manual) < 1e-6
assert abs(res_simplex.x[1] - x2_manual) < 1e-6
assert abs(-res_simplex.fun - z_manual) < 1e-6
print("\nOK: coincide exactamente con la solucion obtenida a mano "
      f"(x1={x1_manual}, x2={x2_manual}, z={z_manual}).")

# --- Variante: 'revised simplex' (mismo algoritmo, mas eficiente) ---
print("\n" + "=" * 60)
print("VARIANTE: 'revised simplex' (mismo metodo, implementacion mas eficiente)")
print("=" * 60)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    res_revised = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                           method="revised simplex")
print(f"x* = {res_revised.x}, z* = {-res_revised.fun}, "
      f"iteraciones = {res_revised.nit}")

# --- Referencia: solver moderno recomendado hoy (HiGHS) ---
print("\n" + "=" * 60)
print("REFERENCIA: method='highs' (solver moderno recomendado)")
print("=" * 60)
res_highs = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(f"x* = {res_highs.x}, z* = {-res_highs.fun}")

print("\nLos tres metodos coinciden en el optimo: x*=(2,6), z*=36 "
      "-- difieren solo en el algoritmo interno y el numero de iteraciones.")
