"""
Comparacion final de los metodos de la Clase 5 en P3 (min (x1-2)^2+(x2-1)^2  s.a. x1^2+x2^2<=1),
con optimo en la frontera curva:  x* = (2,1)/sqrt(5).
Se cuentan evaluaciones de f, de grad f y de restricciones (g y Jg) con contadores.
"""
import os, sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _util import figpath
from _problemas import *
from _functions import rosen_f, rosen_grad, rosen_hess


def cargar(nombre, fichero):
    spec = importlib.util.spec_from_file_location(nombre, os.path.join(HERE, fichero))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


cp = cargar("cp", "12_cutting_plane.py")
sq = cargar("sq", "13_sqp.py")
pe = cargar("pe", "14_penalizacion.py")


class Cuenta:
    """Envuelve una funcion y cuenta cuantas veces se llama."""
    def __init__(self, fn): self.fn, self.n = fn, 0
    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


def instrumentar(f, grad, hess, g, Jg, hessg):
    return dict(f=Cuenta(f), grad=Cuenta(grad), hess=Cuenta(hess), g=Cuenta(g), Jg=Cuenta(Jg), hessg=Cuenta(hessg))


def resumen(nombre, x, xstar, F, C, outer, extra=""):
    viol = max(0.0, float(np.max(C["g"].fn(x))))
    return dict(nombre=nombre, outer=outer, nf=F["f"].n, ng=F["grad"].n, ncons=C["g"].n,
                err=float(np.linalg.norm(x - xstar)), viol=viol, extra=extra)


def correr(prob_name, f, grad, hess, g, Jg, hessg, xstar, x_start, x_feas, hessL, cp_ok=True):
    filas = []
    # ---- SQP (exacta y BFGS)
    for modo in ("exacta", "bfgs"):
        F = instrumentar(f, grad, hess, g, Jg, hessg)
        prob = dict(f=F["f"], grad=F["grad"], hessL=hessL, g=F["g"], Jg=F["Jg"],
                    h=lambda x: np.zeros(0), Jh=lambda x: np.zeros((0, 2)))
        x, lam, mu, hist, nf = sq.sqp(prob, x_start, modo, max_iter=100)
        filas.append(resumen(f"SQP ({'Hessiana exacta' if modo=='exacta' else 'BFGS'})", x, xstar, F, F, len(hist) - 1))
    # ---- cutting plane
    if cp_ok:
        F = instrumentar(f, grad, hess, g, Jg, hessg)
        x, hist, _ = cp.cutting_plane(F["f"], F["grad"], F["g"], F["Jg"], x_start, B=4.0, tol=1e-6, max_iter=200)
        filas.append(resumen("Cutting plane (LP)", x, xstar, F, F, len(hist)))
    # ---- penalizacion exterior / barrera
    for nombre, tipo in (("Penalización exterior", "ext"), ("Barrera interior", "int")):
        F = instrumentar(f, grad, hess, g, Jg, hessg)
        prob = {**pe.NO, **dict(f=F["f"], grad=F["grad"], hess=F["hess"], g=F["g"], Jg=F["Jg"], hessg=F["hessg"])}
        if tipo == "ext":
            x, out = pe.penalizacion_exterior(prob, x_start, [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
        else:
            x, out = pe.barrera_interior(prob, x_feas, [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        filas.append(resumen(nombre, x, xstar, F, F, sum(r["it"] for r in out), extra=f"{len(out)} subproblemas"))
    # ---- scipy
    for metodo in ("SLSQP", "trust-constr"):
        F = instrumentar(f, grad, hess, g, Jg, hessg)
        if metodo == "SLSQP":
            cons = [{"type": "ineq", "fun": lambda z: -F["g"](z), "jac": lambda z: -F["Jg"](z)}]
            r = minimize(F["f"], x_feas, jac=F["grad"], constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
            filas.append(resumen("scipy SLSQP (SQP)", r.x, xstar, F, F, r.nit))
        else:
            nlc = NonlinearConstraint(F["g"], -np.inf, 0, jac=F["Jg"], hess=lambda z, v: sum(vi * H for vi, H in zip(v, F["hessg"](z))))
            r = minimize(F["f"], x_feas, jac=F["grad"], hess=F["hess"], constraints=[nlc], method="trust-constr",
                         options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 500})
            filas.append(resumen("scipy trust-constr (barrera)", r.x, xstar, F, F, r.nit))
    return filas


def imprimir(titulo, filas):
    print(f"\n{titulo}")
    print(f"{'Método':32s} {'iter.':>6} {'f':>6} {'∇f':>6} {'g,∇g':>6} {'||x-x*||':>10} {'viol':>9}")
    for r in filas:
        print(f"{r['nombre']:32s} {r['outer']:>6} {r['nf']:>6} {r['ng']:>6} {r['ncons']:>6} {r['err']:>10.1e} {r['viol']:>9.1e}  {r['extra']}")


def tabla_tex(filas, nombre):
    with open(os.path.join(HERE, "..", "tablas", nombre), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\nM\\'etodo & iter. & eval.\\ $f$ & eval.\\ $\\nabla f$ & $\\|x-x^\\ast\\|$ & viol.\\\\\\midrule\n")
        for r in filas:
            fh.write(f"{r['nombre']} & {r['outer']} & {r['nf']} & {r['ng']} & {r['err']:.0e} & {r['viol']:.0e}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    # ---------------- P3 (convexo)
    hessL3 = lambda x, l, m: hess2(x) + m[0] * hessg3(x)[0]
    filas3 = correr("P3", f2, grad2, hess2, g3, Jg3, hessg3, X3_STAR, np.array([0.3, 0.2]), np.array([0.3, 0.2]), hessL3)
    imprimir("P3: min (x1-2)^2+(x2-1)^2  s.a. x1^2+x2^2<=1   (x* = (2,1)/sqrt5)", filas3)
    tabla_tex(filas3, "tabla_comparacion_P3.tex")

    # ---------------- P4 (no convexo): Rosenbrock en el disco unidad
    x4 = np.array([0.78641515, 0.61769831])
    hessL4 = lambda x, l, m: rosen_hess(x) + m[0] * hessg3(x)[0]
    filas4 = correr("P4", rosen_f, rosen_grad, rosen_hess, g3, Jg3, hessg3, x4, np.array([0.0, 0.5]), np.array([0.0, 0.5]), hessL4, cp_ok=False)
    imprimir("P4: Rosenbrock s.a. x1^2+x2^2<=1  (no convexo; x* = (0.78642, 0.61770))", filas4)
    tabla_tex(filas4, "tabla_comparacion_P4.tex")

    # ---------------- figura: convergencia comparada en P3 (error vs iteracion externa)
    x_start = np.array([0.3, 0.2])   # mismo punto inicial (factible) para todos
    xs_, _, _, hE, _ = sq.sqp(sq.PROB3, x_start, "exacta")
    xs_, _, _, hB, _ = sq.sqp(sq.PROB3, x_start, "bfgs")
    xc, hcp, _ = cp.cutting_plane(f2, grad2, g3, Jg3, np.array([0.3, 0.2]), tol=1e-7, max_iter=200)
    PR = {**pe.NO, **dict(f=f2, grad=grad2, hess=hess2, g=g3, Jg=Jg3, hessg=hessg3)}
    _, oe = pe.penalizacion_exterior(PR, [0.3, 0.2], [10.0**k for k in range(0, 7)])
    _, oi = pe.barrera_interior(PR, [0.3, 0.2], [10.0**(-k) for k in range(0, 6)])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogy([max(np.linalg.norm(h_["x"] - X3_STAR), 1e-16) for h_ in hE], "o-", label="SQP (Hessiana exacta)")
    ax.semilogy([max(np.linalg.norm(h_["x"] - X3_STAR), 1e-16) for h_ in hB], "s--", label="SQP (BFGS)")
    ax.semilogy([max(np.linalg.norm(h_["x"] - X3_STAR), 1e-16) for h_ in hcp], "^-", label="cutting plane")
    ax.semilogy([np.linalg.norm(r["x"] - X3_STAR) for r in oe], "d-", label=r"penalización exterior ($\rho=10^k$)")
    ax.semilogy([np.linalg.norm(r["x"] - X3_STAR) for r in oi], "v-", label=r"barrera interior ($\mu=10^{-k}$)")
    ax.set_xlabel("iteración externa  (para penalización/barrera: $k$ = subproblema)"); ax.set_ylabel(r"$\|x_k-x^\ast\|$")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8); ax.set_title("P3: error frente a iteraciones externas")
    fig.tight_layout(); fig.savefig(figpath("fig15_comparacion_P3.png"), dpi=150); plt.close(fig)
