"""
Los mismos 3 ejemplos, resueltos ahora con PuLP -- una libreria mas
"declarativa" (se escribe el modelo casi como en el papel: variables,
objetivo, restricciones) muy recomendada para ensenar PL/PLE porque no
hay que armar matrices A, b, c a mano como con scipy.

PuLP no resuelve nada por si mismo: por defecto llama al solver libre
CBC (COIN-OR Branch and Cut), que trae empaquetado. CBC tambien hace
branch and bound (con anadidos como "planos de corte", que no vimos en
clase pero son un refinamiento adicional sobre la misma idea).

Instalacion:  pip install pulp
"""
import pulp


def resolver(nombre, coef_obj, restricciones, es_entera):
    """
    coef_obj: dict {"x1": 5, "x2": 4}
    restricciones: lista de (dict_coeficientes, cota_superior)
    es_entera: dict {"x1": True, "x2": False}
    """
    print("=" * 70)
    print(nombre)
    print("=" * 70)

    prob = pulp.LpProblem("mip", pulp.LpMaximize)

    variables = {}
    for nombre_var, entera in es_entera.items():
        cat = "Integer" if entera else "Continuous"
        variables[nombre_var] = pulp.LpVariable(nombre_var, lowBound=0, cat=cat)

    # Funcion objetivo
    prob += pulp.lpSum(coef_obj[v] * variables[v] for v in coef_obj)

    # Restricciones
    for coefs, cota in restricciones:
        prob += pulp.lpSum(coefs[v] * variables[v] for v in coefs) <= cota

    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # msg=0: silenciar el log de CBC

    print("Estado:", pulp.LpStatus[prob.status])
    for v in variables.values():
        print(f"  {v.name} = {v.value()}")
    print(f"  z* = {pulp.value(prob.objective)}")
    print()
    return {v: variables[v].value() for v in variables}, pulp.value(prob.objective)


x1, z1 = resolver(
    "EJEMPLO 1: max 5x1+4x2 s.a. 6x1+4x2<=24, x1+2x2<=6, x1,x2 enteros",
    coef_obj={"x1": 5, "x2": 4},
    restricciones=[({"x1": 6, "x2": 4}, 24), ({"x1": 1, "x2": 2}, 6)],
    es_entera={"x1": True, "x2": True},
)

x2, z2 = resolver(
    "EJEMPLO 2: max 5x1+3x2 s.a. 2x1+3x2<=12, 3x1+x2<=9, x1,x2 enteros",
    coef_obj={"x1": 5, "x2": 3},
    restricciones=[({"x1": 2, "x2": 3}, 12), ({"x1": 3, "x2": 1}, 9)],
    es_entera={"x1": True, "x2": True},
)

x3, z3 = resolver(
    "EJEMPLO 3 (MIXTO): max 8x1+5x2 s.a. x1+x2<=6, 9x1+5x2<=45, x1 entera, x2 continua",
    coef_obj={"x1": 8, "x2": 5},
    restricciones=[({"x1": 1, "x2": 1}, 6), ({"x1": 9, "x2": 5}, 45)],
    es_entera={"x1": True, "x2": False},
)

assert (x1["x1"], x1["x2"], z1) == (4, 0, 20)
assert (x2["x1"], x2["x2"], z2) == (2, 2, 16)
assert x3["x1"] == 4 and abs(x3["x2"] - 1.8) < 1e-6 and z3 == 41
print("OK: los 3 resultados de PuLP/CBC coinciden con scipy.optimize.milp "
      "y con el branch & bound hecho a mano.")
