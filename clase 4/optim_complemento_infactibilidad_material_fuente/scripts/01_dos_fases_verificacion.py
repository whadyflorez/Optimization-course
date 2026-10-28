"""
Metodo de Dos Fases, verificado con fracciones exactas.
Convencion (identica a toda la Clase 4): la fila objetivo se guarda como
"z - C^T x = rhs" con C_j = -c_j (coeficientes ALMACENADOS son el negativo
de los coeficientes reales del objetivo). Criterio de entrada: coeficiente
ALMACENADO mas NEGATIVO (equivale a mayor c_j real, en maximizacion).

EJEMPLO A (factible tras Fase 1):
    max z = 2x1+3x2  s.a.  x1+x2<=4,  x1+2x2>=4,  x1,x2>=0

EJEMPLO B (infactible, el nodo del arbol B&B de la Clase 5):
    2x1+3x2<=12,  x1>=2 (con x1<=2 tambien, cota superior),  x2>=3
"""
from fractions import Fraction as F


class Tableau:
    def __init__(self, names, rows, basis, zrow):
        self.names = names
        self.rows = dict(rows)
        self.basis = list(basis)
        self.zrow = zrow

    def print_row(self, label, coeffs, rhs):
        terms = " + ".join(f"{coeffs[j]}*{self.names[j]}" for j in range(len(self.names)) if coeffs[j] != 0)
        print(f"  {label}: {terms} = {rhs}")

    def show(self, ztitle="z"):
        for n in self.basis:
            self.print_row(n, *self.rows[n])
        self.print_row(ztitle, *self.zrow)

    def eliminate_basics_from_z(self):
        zc, zr = self.zrow
        for n in self.basis:
            idx = self.names.index(n)
            factor = zc[idx]
            if factor != 0:
                pc, prhs = self.rows[n]
                zc = [c - factor * p for c, p in zip(zc, pc)]
                zr = zr - factor * prhs
        self.zrow = (zc, zr)

    def pivot(self, entering_idx, leaving_name):
        piv_coeffs, piv_rhs = self.rows[leaving_name]
        piv_val = piv_coeffs[entering_idx]
        new_piv = [c / piv_val for c in piv_coeffs]
        new_piv_rhs = piv_rhs / piv_val
        self.rows[leaving_name] = (new_piv, new_piv_rhs)
        for name in list(self.rows.keys()):
            if name == leaving_name:
                continue
            coeffs, rhs = self.rows[name]
            factor = coeffs[entering_idx]
            if factor != 0:
                self.rows[name] = (
                    [c - factor * npc for c, npc in zip(coeffs, new_piv)],
                    rhs - factor * new_piv_rhs,
                )
        zc, zr = self.zrow
        factor = zc[entering_idx]
        if factor != 0:
            zc = [c - factor * npc for c, npc in zip(zc, new_piv)]
            zr = zr - factor * new_piv_rhs
        self.zrow = (zc, zr)
        self.basis = [self.names[entering_idx] if b == leaving_name else b for b in self.basis]
        self.rows[self.names[entering_idx]] = self.rows.pop(leaving_name)

    def run_simplex(self, exclude_prefix=None, label="z"):
        it = 0
        while True:
            it += 1
            zc, zr = self.zrow
            cand = {j: zc[j] for j in range(len(self.names)) if zc[j] < 0
                    and not (exclude_prefix and self.names[j].startswith(exclude_prefix))}
            if not cand:
                print(f"--- {label}, iteracion {it}: sin coeficientes negativos -> OPTIMO ---")
                break
            entering = min(cand, key=cand.get)
            ratios = {}
            for n in self.basis:
                c, r = self.rows[n]
                if c[entering] > 0:
                    ratios[n] = r / c[entering]
            if not ratios:
                print(f"{label}: NO ACOTADO al entrar {self.names[entering]}")
                break
            leaving = min(ratios, key=ratios.get)
            print(f"{label}, iteracion {it}: entra {self.names[entering]}, sale {leaving}  "
                  f"(razones: {{{', '.join(f'{k}:{v}' for k,v in ratios.items())}}})")
            self.pivot(entering, leaving)
            self.show(label)
            print()


print("#" * 72)
print("EJEMPLO A: max z=2x1+3x2 s.a. x1+x2<=4, x1+2x2>=4")
print("#" * 72)
names = ["x1", "x2", "s1", "s2", "a1"]
rows = {
    "s1": ([F(1), F(1), F(1), F(0), F(0)], F(4)),
    "a1": ([F(1), F(2), F(0), F(-1), F(1)], F(4)),
}
basis = ["s1", "a1"]
wrow = ([F(0), F(0), F(0), F(0), F(1)], F(0))  # w'=-a1 -> c_a1=-1 -> stored=+1
t = Tableau(names, rows, basis, wrow)
t.eliminate_basics_from_z()
print("Tableau inicial de Fase 1:")
t.show("w'")
print()
t.run_simplex(exclude_prefix="a", label="Fase 1")
w = -t.zrow[1]
print(f"w = {w} -> {'FACTIBLE, pasar a Fase 2' if w == 0 else 'INFACTIBLE'}\n")

# Fase 2: descartar columna a1, restaurar z real
names2 = ["x1", "x2", "s1", "s2"]
rows2 = {n: (t.rows[n][0][:4], t.rows[n][1]) for n in t.basis}
zrow2 = ([F(-2), F(-3), F(0), F(0)], F(0))  # z=2x1+3x2 -> stored=-c_j
t2 = Tableau(names2, rows2, t.basis, zrow2)
t2.eliminate_basics_from_z()
print("Tableau inicial de Fase 2 (se descarta a1, se restaura z real):")
t2.show("z")
print()
t2.run_simplex(label="Fase 2")
print(f"\n>>> OPTIMO EJEMPLO A: z* = {t2.zrow[1]}")
sol = {n: F(0) for n in names2}
for n in t2.basis:
    sol[n] = t2.rows[n][1]
print("x* =", {k: str(v) for k, v in sol.items() if k in ["x1", "x2"]})


print("\n" + "#" * 72)
print("EJEMPLO B (INFACTIBLE): 2x1+3x2<=12, x1>=2, x2>=3  (nodo del arbol B&B, Clase 5)")
print("#" * 72)
namesB = ["x1", "x2", "s1", "s2", "a1", "s3", "a2"]
rowsB = {
    "s1": ([F(2), F(3), F(1), F(0), F(0), F(0), F(0)], F(12)),
    "a1": ([F(1), F(0), F(0), F(-1), F(1), F(0), F(0)], F(2)),
    "a2": ([F(0), F(1), F(0), F(0), F(0), F(-1), F(1)], F(3)),
}
basisB = ["s1", "a1", "a2"]
wrowB = ([F(0), F(0), F(0), F(0), F(1), F(0), F(1)], F(0))
tb = Tableau(namesB, rowsB, basisB, wrowB)
tb.eliminate_basics_from_z()
print("Tableau inicial de Fase 1:")
tb.show("w'")
print()
tb.run_simplex(exclude_prefix="a", label="Fase 1")
wB = -tb.zrow[1]
print(f"\nw = a1+a2 = {wB}  ->  {'FACTIBLE' if wB == 0 else 'INFACTIBLE: no existe solucion'}")
