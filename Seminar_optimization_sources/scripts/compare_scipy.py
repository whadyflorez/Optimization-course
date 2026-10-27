"""Manual implementations vs scipy.optimize (iterations / evaluations)."""
import json
import numpy as np
from scipy.optimize import (minimize, minimize_scalar, dual_annealing, NonlinearConstraint)
import problems as P, methods as M


class Count:
    def __init__(self, fn): self.fn, self.n = fn, 0
    def __call__(self, x):
        self.n += 1
        return self.fn(x)


out = {}
# ---- P1 golden section
F = Count(P.cost_insulation)
t, hist = M.golden_section(F, 0.0, 0.15, 1e-4)
out["golden"] = dict(manual=dict(x=t, f=P.cost_insulation(np.array([t])), it=len(hist) - 1, nf=F.n))
r = minimize_scalar(lambda v: P.cost_insulation(np.array([v])), bracket=(0.0, 0.05, 0.15), method="golden", options=dict(xtol=1e-4))
out["golden"]["scipy"] = dict(x=r.x, f=r.fun, it=r.nit, nf=r.nfev)

# ---- P3 gradient-based and simplex
X0 = np.array([0.5, 0.5]); f = P.cost_exchanger
res = {}
F = Count(f); x, p = M.gradient_descent(F, X0, "line_search", n_iter=2000); res["GD + line search"] = dict(it=len(p) - 1, nf=F.n, f=f(x))
F = Count(f); x, p = M.newton(F, X0); res["Newton"] = dict(it=len(p) - 1, nf=F.n, f=f(x))
F = Count(f); x, p, _ = M.trust_region_dogleg(F, X0); res["Dogleg"] = dict(it=len(p) - 1, nf=F.n, f=f(x))
F = Count(f); x, fb, h = M.nelder_mead(F, X0.copy(), 0.5, tol=1e-9); res["Nelder-Mead"] = dict(it=len(h) - 1, nf=F.n, f=fb)
out["P3_manual"] = res
sp = {}
F = Count(f); r = minimize(F, X0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9)); sp["Nelder-Mead"] = dict(it=r.nit, nf=r.nfev, f=r.fun)
F = Count(f); r = minimize(F, X0, method="BFGS"); sp["BFGS"] = dict(it=r.nit, nf=r.nfev, f=r.fun)
gr = lambda x: M.numgrad(f, x); hs = lambda x: M.numhess(f, x)
r = minimize(f, X0, jac=gr, hess=hs, method="dogleg"); sp["dogleg (needs H > 0)"] = dict(it=r.nit, nf=r.nfev, f=r.fun)
r = minimize(f, X0, jac=gr, hess=hs, method="trust-exact"); sp["trust-exact"] = dict(it=r.nit, nf=r.nfev, f=r.fun)
r = minimize(f, X0, jac=gr, hess=hs, method="Newton-CG"); sp["Newton-CG"] = dict(it=r.nit, nf=r.nfev, f=r.fun)
out["P3_scipy"] = sp

# ---- P2 stochastic
f2 = P.temperature_board; lo, hi = P.LO2, P.HI2
bx, bf, *_ = M.simulated_annealing(f2, np.array([23.0, 8.0]), 20.0, 1 - 6 / 2000, 1.0, 2000, np.random.default_rng(0), lo, hi)
r = dual_annealing(f2, list(zip(lo, hi)), x0=np.array([23.0, 8.0]), seed=0, maxfun=2000)
out["P2"] = dict(manual=dict(x=bx.tolist(), f=bf), scipy=dict(x=r.x.tolist(), f=r.fun, nf=r.nfev))

# ---- P4 constrained
c = np.array([5.0, -2.0]); lo4, hi4 = np.array([0.0, 1.0]), np.array([6.0, 9.0])
xcp, path, _ = M.cutting_plane(c, P.g_solar, [0, 0], lo4, hi4, tol=1e-4)
r = minimize(P.cost_solar, [3.0, 3.0], constraints=[{"type": "ineq", "fun": lambda x: -P.g_solar(x)}], bounds=list(zip(lo4, hi4)), method="SLSQP", options=dict(ftol=1e-12))
nlc = NonlinearConstraint(P.g_solar, -np.inf, 0.0)
r2 = minimize(P.cost_solar, [4.0, 3.0], constraints=[nlc], bounds=list(zip(lo4, hi4)), method="trust-constr")
out["P4"] = dict(cutting=dict(x=xcp.tolist(), f=P.cost_solar(xcp), it=len(path) - 1),
                 slsqp=dict(x=r.x.tolist(), f=r.fun, it=r.nit), trust_constr=dict(x=r2.x.tolist(), f=r2.fun, it=r2.nit))
json.dump(out, open("../tablas/compare_scipy.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
