"""
Rastreador de Branch & Bound para problemas de 2 variables (con posibilidad
de declarar solo un subconjunto como enteras -> soporta MIP mixto).
Usa scipy.optimize.linprog en cada nodo. Imprime el arbol completo con
poda explicita, para poder disenar a mano ejemplos pedagogicos con y sin
poda antes de escribirlos en las diapositivas.
"""
from scipy.optimize import linprog
import itertools

TOL = 1e-6


def solve_node(c, A_ub, b_ub, bounds):
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res


def is_int(v):
    return abs(v - round(v)) < TOL


def branch_and_bound(c, A_ub, b_ub, bounds, integer_vars, maximize=True,
                      verbose=True):
    """integer_vars: set de indices (0-based) que deben ser enteros."""
    sign = -1 if maximize else 1
    c_arr = [sign * ci for ci in c]

    best = {"z": -float("inf") if maximize else float("inf"), "x": None}
    node_id = [0]

    def rec(A_ub, b_ub, bounds, depth, label):
        node_id[0] += 1
        nid = node_id[0]
        res = solve_node(c_arr, A_ub, b_ub, bounds)
        indent = "  " * depth
        if not res.success:
            if verbose:
                print(f"{indent}[Nodo {nid}] {label}: INFACTIBLE -> poda")
            return
        z = sign * res.fun
        x = res.x
        if verbose:
            print(f"{indent}[Nodo {nid}] {label}: x={x.round(4)}  z={z:.4f}")

        # poda por cota
        if maximize and z <= best["z"] + TOL and best["x"] is not None:
            if verbose:
                print(f"{indent}   -> cota {z:.4f} <= incumbente {best['z']:.4f}: PODA (por cota)")
            return
        if (not maximize) and z >= best["z"] - TOL and best["x"] is not None:
            if verbose:
                print(f"{indent}   -> cota {z:.4f} >= incumbente {best['z']:.4f}: PODA (por cota)")
            return

        # buscar variable entera fraccionaria
        frac_idx = None
        for i in integer_vars:
            if not is_int(x[i]):
                frac_idx = i
                break

        if frac_idx is None:
            # solucion valida para el MIP
            better = (z > best["z"] + TOL) if maximize else (z < best["z"] - TOL)
            if better:
                best["z"] = z
                best["x"] = x
                if verbose:
                    print(f"{indent}   -> ENTERA VALIDA. Nuevo incumbente z={z:.4f}")
            else:
                if verbose:
                    print(f"{indent}   -> ENTERA VALIDA pero no mejora incumbente")
            return

        # ramificar
        val = x[frac_idx]
        floor_v, ceil_v = int(val // 1), int(val // 1) + 1
        # hijo izquierdo: x_i <= floor
        row = [0] * len(c)
        row[frac_idx] = 1
        A_L = A_ub + [row]
        b_L = b_ub + [floor_v]
        rec(A_L, b_L, bounds, depth + 1, f"x{frac_idx+1}<={floor_v}")
        # hijo derecho: x_i >= ceil  ->  -x_i <= -ceil
        row2 = [0] * len(c)
        row2[frac_idx] = -1
        A_R = A_ub + [row2]
        b_R = b_ub + [-ceil_v]
        rec(A_R, b_R, bounds, depth + 1, f"x{frac_idx+1}>={ceil_v}")

    rec(A_ub, b_ub, bounds, 0, "raiz (relajacion LP)")
    print(f"\n>>> OPTIMO MIP: x={best['x'].round(4)}  z={best['z']:.4f}\n")
    return best


if __name__ == "__main__":
    print("=" * 70)
    print("EJEMPLO 1 (ya visto en clase): max 5x1+4x2, integer_vars={0,1}")
    print("=" * 70)
    branch_and_bound(
        c=[5, 4], A_ub=[[6, 4], [1, 2]], b_ub=[24, 6],
        bounds=[(0, None), (0, None)], integer_vars={0, 1},
    )
