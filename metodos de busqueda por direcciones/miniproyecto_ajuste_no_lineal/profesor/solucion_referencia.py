"""
SOLUCION DE REFERENCIA (SOLO PROFESOR)
Ajuste no lineal de minimos cuadrados  y(t) = a e^{-bt} + c e^{-dt}
con tres minimizadores implementados a mano: Dogleg, Levenberg-Marquardt y SGD,
comparados con scipy.optimize.least_squares.

Costo:  f(th) = 0.5 * sum_i r_i(th)^2 ,   r_i = modelo(t_i; th) - y_i
Gradiente  g = J^T r ,  Hessiana de Gauss-Newton  B = J^T J.
"""
import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generar_datos import generar


# ------------------------------------------------------------------ modelo
def modelo(th, t):
    a, b, c, d = th
    return a * np.exp(-b * t) + c * np.exp(-d * t)


def residuos(th, t, y):
    return modelo(th, t) - y


def jacobiana(th, t):
    a, b, c, d = th
    e1, e2 = np.exp(-b * t), np.exp(-d * t)
    return np.column_stack([e1, -a * t * e1, e2, -c * t * e2])


def costo(th, t, y):
    r = residuos(th, t, y)
    return 0.5 * r @ r


# ------------------------------------------------------------------ dogleg
def paso_dogleg(g, B, Delta):
    """Paso dogleg para el modelo m(p) = g.p + 0.5 p'Bp, ||p|| <= Delta."""
    gBg = g @ B @ g
    if gBg <= 0:
        pC = -Delta / np.linalg.norm(g) * g
    else:
        pU = -(g @ g) / gBg * g
        pC = pU if np.linalg.norm(pU) < Delta else -Delta / np.linalg.norm(g) * g
    # el dogleg exige B definida positiva; si no, solo Cauchy (siempre reduce el modelo)
    w = np.linalg.eigvalsh(B)
    if w.min() <= 1e-12 * w.max():
        return pC
    pN = np.linalg.solve(B, -g)
    if np.linalg.norm(pN) <= Delta:
        return pN
    if np.linalg.norm(pC) >= Delta:
        return Delta / np.linalg.norm(pC) * pC
    d = pN - pC
    a, b, c = d @ d, 2 * pC @ d, pC @ pC - Delta**2
    tau = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return pC + tau * d


def dogleg_ls(th0, t, y, Delta0=1.0, Delta_max=100.0, tol=1e-6, max_iter=200):
    th = np.array(th0, float)
    Delta = Delta0
    hist = [th.copy()]
    costs = [costo(th, t, y)]
    nfev = 1
    for k in range(max_iter):
        r, J = residuos(th, t, y), jacobiana(th, t)
        g, B = J.T @ r, J.T @ J
        if np.linalg.norm(g, np.inf) < tol:
            break
        p = paso_dogleg(g, B, Delta)
        pred = -(g @ p + 0.5 * p @ B @ p)
        f_new = costo(th + p, t, y)
        nfev += 1
        rho = (costs[-1] - f_new) / pred if pred > 1e-16 else -1.0
        if rho > 0:
            th = th + p
        if rho < 0.25:
            Delta *= 0.25
        elif rho > 0.75 and np.isclose(np.linalg.norm(p), Delta, rtol=1e-6):
            Delta = min(2 * Delta, Delta_max)
        hist.append(th.copy())
        costs.append(costo(th, t, y))
    return np.array(hist), np.array(costs), nfev, k + 1


# ------------------------------------------------------- Levenberg-Marquardt
def lm_ls(th0, t, y, lam0=1e-2, tol=1e-6, max_iter=200):
    """LM con escalado de Marquardt (diag(J'J)) y actualizacion de lambda de Nielsen."""
    th = np.array(th0, float)
    lam, nu = lam0, 2.0
    hist = [th.copy()]
    costs = [costo(th, t, y)]
    nfev = 1
    for k in range(max_iter):
        r, J = residuos(th, t, y), jacobiana(th, t)
        g, B = J.T @ r, J.T @ J
        if np.linalg.norm(g, np.inf) < tol:
            break
        A = B + lam * np.diag(np.diag(B))
        p = np.linalg.solve(A, -g)
        pred = -(g @ p + 0.5 * p @ B @ p)
        f_new = costo(th + p, t, y)
        nfev += 1
        if np.linalg.norm(p) < 1e-10 * (1 + np.linalg.norm(th)):
            break                                     # paso despreciable: convergio
        rho = (costs[-1] - f_new) / pred if pred > 1e-16 else -1.0
        if rho > 0:                                   # paso aceptado
            th = th + p
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:                                         # rechazado: mas amortiguado
            lam *= nu
            nu *= 2
        hist.append(th.copy())
        costs.append(costo(th, t, y))
    return np.array(hist), np.array(costs), nfev, k + 1


# ---------------------------------------------------------------------- SGD
def sgd_ls(th0, t, y, batch=16, lr0=0.02, k0=2000, epocas=300, seed=0, beta=0.0):
    """SGD con minilotes y paso decreciente lr_k = lr0 / (1 + k/k0).
    Se minimiza el costo medio F = f/n; el gradiente del lote es J_B' r_B / |B|."""
    rng = np.random.default_rng(seed)
    th = np.array(th0, float)
    n = len(t)
    hist, costs = [th.copy()], [costo(th, t, y)]
    k = 0
    v = np.zeros_like(th)
    for ep in range(epocas):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            r = residuos(th, t[idx], y[idx])
            g = jacobiana(th, t[idx]).T @ r / len(idx)
            v = beta * v - lr0 / (1 + k / k0) * g        # beta=0: SGD puro
            th = th + v
            k += 1
        hist.append(th.copy())                       # una entrada por epoca
        costs.append(costo(th, t, y))
    return np.array(hist), np.array(costs), k


# ---------------------------------------------------------------- estadistica
def incertidumbre(th, t, y):
    J = jacobiana(th, t)
    n, p = J.shape
    s2 = 2 * costo(th, t, y) / (n - p)
    cov = s2 * np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    return s2, sd, corr, np.linalg.cond(J.T @ J)


if __name__ == "__main__":
    import json
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")

    ID = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    carpeta = os.path.dirname(os.path.abspath(__file__))
    figs = os.path.join(carpeta, "figs")
    os.makedirs(figs, exist_ok=True)
    t, y, th_true = generar(ID)
    n = len(t)
    th0 = np.array([1.0, 1.0, 1.0, 0.5])
    R = {"id": ID, "n": n, "th_true": th_true.tolist(), "th0": th0.tolist()}

    # ---- referencia (scipy, LM de MINPACK, muy ajustado)
    ref = least_squares(residuos, th0, jac=lambda th, t, y: jacobiana(th, t),
                        args=(t, y), method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15)
    th_star, f_star = ref.x, 0.5 * ref.fun @ ref.fun
    R.update(th_star=th_star.tolist(), f_star=f_star)
    rel = lambda f: max((f - f_star) / f_star, 1e-16)

    # ================= E1: comparacion de los tres metodos (inicio A)
    filas = []
    trazas = {}
    for nombre, fun in (("Dogleg", dogleg_ls), ("Levenberg--Marquardt", lm_ls)):
        t0 = time.perf_counter()
        H, C, nfev, nit = fun(th0, t, y)
        dt = time.perf_counter() - t0
        trazas[nombre] = (H, C)
        filas.append(dict(m=nombre, it=nit, ev=2 * n * nit, f=C[-1], err=np.linalg.norm(H[-1] - th_star), t=dt * 1e3))
    t0 = time.perf_counter()
    rs = least_squares(residuos, th0, jac=lambda th, t, y: jacobiana(th, t), args=(t, y), method="lm")
    dt = time.perf_counter() - t0
    filas.append(dict(m="scipy least\\_squares (lm)", it=rs.nfev, ev=2 * n * rs.nfev, f=0.5 * rs.fun @ rs.fun,
                      err=np.linalg.norm(rs.x - th_star), t=dt * 1e3))
    rs2 = least_squares(residuos, th0, jac=lambda th, t, y: jacobiana(th, t), args=(t, y), method="trf")
    filas.append(dict(m="scipy least\\_squares (trf)", it=rs2.nfev, ev=2 * n * rs2.nfev, f=0.5 * rs2.fun @ rs2.fun,
                      err=np.linalg.norm(rs2.x - th_star), t=np.nan))

    sgd_cfg = {
        "SGD puro, lote 16": dict(batch=16, lr0=0.1, k0=5000, epocas=5000),
        "SGD + momentum 0.9, lote 16": dict(batch=16, lr0=0.01, k0=20000, epocas=1500, beta=0.9),
    }
    sgd_res = {}
    for nombre, cfg in sgd_cfg.items():
        t0 = time.perf_counter()
        H, C, k = sgd_ls(th0, t, y, **cfg)
        dt = time.perf_counter() - t0
        sgd_res[nombre] = (H, C, k, cfg)
        filas.append(dict(m=nombre, it=k, ev=k * cfg["batch"], f=C[-1], err=np.linalg.norm(H[-1] - th_star), t=dt * 1e3))
    R["E1"] = filas
    with open(os.path.join(carpeta, "tabla_E1.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        fh.write("M\\'etodo & Iter./pasos & Eval. de datos & $f(\\theta)$ & $\\|\\theta-\\theta^\\ast\\|$ & ms \\\\\\midrule\n")
        for r in filas:
            ms = "--" if np.isnan(r["t"]) else f"{r['t']:.0f}"
            miles = lambda v: f"{v:,}".replace(",", "\\,")
            fh.write(f"{r['m']} & {miles(r['it'])} & {miles(r['ev'])} & {r['f']:.5f} & {r['err']:.1e} & {ms} \\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")

    # ---- Fig 1: ajuste
    tt = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(t, y, ".", color="gray", ms=5, label="datos")
    ax[0].plot(tt, modelo(th_star, tt), color="crimson", lw=2, label="ajuste")
    ax[0].plot(tt, th_star[0] * np.exp(-th_star[1] * tt), "--", color="tab:blue", lw=1.2, label="componente rápida")
    ax[0].plot(tt, th_star[2] * np.exp(-th_star[3] * tt), "--", color="tab:green", lw=1.2, label="componente lenta")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Datos y ajuste biexponencial")
    ax[1].plot(t, -ref.fun, ".", color="gray", ms=5)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("residuo"); ax[1].set_title("Residuos del ajuste"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig_ajuste.png"), dpi=150); plt.close(fig)

    # ---- Fig 2: convergencia
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for nombre, (H, C) in trazas.items():
        ax[0].semilogy([rel(c) for c in C], "o-", ms=4, label=nombre)
    ax[0].set_xlabel("iteración"); ax[0].set_ylabel(r"$(f-f^\ast)/f^\ast$"); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title("Dogleg y LM"); ax[0].legend()
    for nombre, (H, C, k, cfg) in sgd_res.items():
        ax[1].semilogy([rel(c) for c in C], label=nombre)
    ax[1].axhline(0.05, ls="--", color="gray"); ax[1].text(5, 0.06, "meta: 5 %", color="gray")
    ax[1].set_xlabel("época (pasada completa por los datos)"); ax[1].set_title("Gradiente estocástico")
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig_convergencia.png"), dpi=150); plt.close(fig)

    # ================= E2: SGD, efecto del lote y de la tasa de aprendizaje
    cfgs = [("lote 150 (gradiente completo)", dict(batch=150, lr0=0.1, k0=5000, epocas=3000)),
            ("lote 16", dict(batch=16, lr0=0.1, k0=5000, epocas=3000)),
            ("lote 4", dict(batch=4, lr0=0.1, k0=5000, epocas=3000)),
            ("lote 16, paso constante 0.1", dict(batch=16, lr0=0.1, k0=1e12, epocas=3000)),
            ("lote 16, paso constante 0.02", dict(batch=16, lr0=0.02, k0=1e12, epocas=3000))]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    E2 = []
    for i, (nombre, cfg) in enumerate(cfgs):
        H, C, k = sgd_ls(th0, t, y, **cfg)
        a = ax[0] if i < 3 else ax[1]
        a.semilogy([rel(c) for c in C], label=nombre)
        E2.append(dict(m=nombre, pasos=k, rel=rel(C[-1])))
    ax[0].set_title("Efecto del tamaño de lote (paso decreciente)")
    ax[1].set_title("Paso constante vs. decreciente")
    ax[1].semilogy([rel(c) for c in sgd_ls(th0, t, y, **cfgs[1][1])[1]], "--", color="k", label="lote 16, decreciente")
    for a in ax:
        a.axhline(0.05, ls="--", color="gray"); a.set_xlabel("época"); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    ax[0].set_ylabel(r"$(f-f^\ast)/f^\ast$")
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig_sgd_parametros.png"), dpi=150); plt.close(fig)
    R["E2"] = E2

    # ================= E3: sensibilidad al punto inicial
    inits = [("A", [1, 1, 1, 0.5]), ("B", [5, 5, 5, 5]), ("C", [1, 0.3, 1, 0.35]),
             ("D", [1, 0.1, 1, 2]), ("E", [3, 0.5, 0.5, 0.05]), ("F", [1, 1, 1, 1])]
    E3 = []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
    for nm, th_i in inits:
        th_i = np.array(th_i, float)
        row = dict(nom=nm, th0=th_i.tolist())
        for key, fun in (("dog", dogleg_ls), ("lm", lm_ls)):
            H, C, nfev, nit = fun(th_i, t, y, max_iter=300)
            row[key] = dict(it=nit, f=C[-1], th=H[-1].tolist())
            if nm in ("A", "F", "D"):
                a = ax[0] if key == "dog" else ax[1]
                a.plot(H[:, 1], H[:, 3], "o-", ms=3, lw=1, label=f"inicio {nm}")
        E3.append(row)
    for a, ttl in zip(ax, ("Dogleg", "Levenberg–Marquardt")):
        a.plot(*th_star[[1, 3]], "*", color="crimson", ms=15, label=r"$\theta^\ast$ (b,d)", zorder=5)
        a.plot(*th_star[[3, 1]], "*", color="orange", ms=15, label="solución permutada", zorder=5)
        a.set_xlabel("b"); a.set_title(ttl); a.grid(alpha=.3); a.set_xlim(0, 2.2); a.set_ylim(0, 2.2)
    ax[0].set_ylabel("d"); ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(figs, "fig_trayectorias_bd.png"), dpi=150); plt.close(fig)
    R["E3"] = E3
    with open(os.path.join(carpeta, "tabla_E3.tex"), "w") as fh:
        fh.write("\\begin{tabular}{llrrrr}\\toprule\n")
        fh.write("& $\\theta_0$ & \\multicolumn{2}{c}{Dogleg} & \\multicolumn{2}{c}{LM} \\\\\n")
        fh.write("& & iter. & $f_{\\rm final}$ & iter. & $f_{\\rm final}$ \\\\\\midrule\n")
        for r in E3:
            th_txt = "(" + ", ".join(f"{v:g}" for v in r["th0"]) + ")"
            fh.write(f"{r['nom']} & ${th_txt}$ & {r['dog']['it']} & {r['dog']['f']:.4f} & {r['lm']['it']} & {r['lm']['f']:.4f} \\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")

    # ================= E4: incertidumbre y ruido
    s2, sd, corr, kappa = incertidumbre(th_star, t, y)
    R["E4"] = dict(sigma_hat=float(np.sqrt(s2)), sd=sd.tolist(), corr=corr.tolist(), cond=float(kappa))
    ruido = []
    for sig in (0.05, 0.2, 0.5):
        tt_, yy_, _ = generar(ID, sigma=sig)
        r_ = least_squares(residuos, th_star, jac=lambda th, t, y: jacobiana(th, t), args=(tt_, yy_), method="lm")
        _, sd_, _, _ = incertidumbre(r_.x, tt_, yy_)
        ruido.append(dict(sigma=sig, th=r_.x.tolist(), sd=sd_.tolist()))
    R["ruido"] = ruido

    json.dump(R, open(os.path.join(carpeta, "resultados_referencia.json"), "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))
