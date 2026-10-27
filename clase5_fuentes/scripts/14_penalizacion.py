"""
Metodos de penalizacion:
  EXTERIOR (penalizacion cuadratica):  Q(x;rho) = f + rho/2 * ( sum h^2 + sum max(0,g)^2 ),   rho -> inf
  INTERIOR (barrera logaritmica):      B(x;mu)  = f - mu * sum log(-g),                        mu  -> 0
En ambos casos se resuelve una SECUENCIA de problemas SIN restricciones (con BFGS, de la Clase 4),
usando como punto inicial la solucion del problema anterior (warm start).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _problemas import *


# ------------------------------------------------------- minimizador sin restricciones
def newton(F, gradF, hessF, x0, tol=1e-10, max_iter=100):
    """Newton amortiguado (Clase 4) con busqueda lineal de Armijo.  F puede devolver +inf fuera
    de su dominio (barrera): la busqueda lineal retrocede hasta volver a un punto valido.
    Si la Hessiana no es definida positiva se le suma un multiplo de la identidad."""
    x = np.array(x0, float); n = len(x)
    fx = F(x); nf = 1
    for k in range(max_iter):
        g = gradF(x)
        if np.linalg.norm(g, np.inf) < tol:
            return x, k, nf
        H = hessF(x)
        wmin = np.linalg.eigvalsh(H).min()
        if wmin < 1e-10:
            H = H + (1e-10 - wmin + 1e-6) * np.eye(n)
        p = -np.linalg.solve(H, g)
        a = 1.0
        while True:
            xn = x + a * p; fn = F(xn); nf += 1
            if np.isfinite(fn) and fn <= fx + 1e-4 * a * (g @ p) + 1e-14 * (1 + abs(fx)):   # + tolerancia de redondeo
                break
            a *= 0.5
            if a < 1e-14:
                return x, k, nf
        x, fx = xn, fn
    return x, max_iter, nf


# ----------------------------------------------------------------- exterior
def penalizacion_exterior(prob, x0, rhos):
    """prob: dict con f, grad, hess, h, Jh, hessh, g, Jg, hessg  (h o g pueden ser vacias)."""
    f, grad, hess = prob["f"], prob["grad"], prob["hess"]
    h, Jh, hessh, g, Jg, hessg = prob["h"], prob["Jh"], prob["hessh"], prob["g"], prob["Jg"], prob["hessg"]
    x = np.array(x0, float); out = []
    for rho in rhos:
        def Q(z):
            v = f(z)
            if len(h(z)): v += 0.5 * rho * np.sum(h(z)**2)
            if len(g(z)): v += 0.5 * rho * np.sum(np.maximum(0, g(z))**2)
            return v
        def dQ(z):
            v = grad(z).copy()
            if len(h(z)): v += rho * Jh(z).T @ h(z)
            if len(g(z)): v += rho * Jg(z).T @ np.maximum(0, g(z))
            return v
        def HQ(z):
            M = hess(z).copy()
            if len(h(z)):
                M += rho * (Jh(z).T @ Jh(z) + sum(hi * Hi for hi, Hi in zip(h(z), hessh(z))))
            if len(g(z)):
                for gi, gr, Hi in zip(g(z), Jg(z), hessg(z)):
                    if gi > 0: M += rho * (np.outer(gr, gr) + gi * Hi)
            return M
        x, it, nf = newton(Q, dQ, HQ, x, tol=1e-9 * max(1.0, rho))
        w = np.linalg.eigvalsh(HQ(x))
        out.append(dict(rho=rho, x=x.copy(), f=f(x), it=it, nf=nf, cond=w.max() / max(w.min(), 1e-300),
                        viol=max([0.0] + list(np.abs(h(x))) + list(np.maximum(0, g(x)))),
                        lam=rho * h(x) if len(h(x)) else np.zeros(0),
                        mu=rho * np.maximum(0, g(x)) if len(g(x)) else np.zeros(0)))
    return x, out


# ----------------------------------------------------------------- interior
def barrera_interior(prob, x0, mus):
    f, grad, hess, g, Jg, hessg = prob["f"], prob["grad"], prob["hess"], prob["g"], prob["Jg"], prob["hessg"]
    x = np.array(x0, float); out = []
    assert np.all(g(x) < 0), "x0 debe ser estrictamente factible"
    for mu in mus:
        def B(z):
            gz = g(z)
            if np.any(gz >= 0): return np.inf
            return f(z) - mu * np.sum(np.log(-gz))
        def dB(z):
            return grad(z) - mu * Jg(z).T @ (1.0 / g(z))
        def HB(z):
            M = hess(z).copy()
            for gi, gr, Hi in zip(g(z), Jg(z), hessg(z)):
                M += mu * (np.outer(gr, gr) / gi**2 - Hi / gi)
            return M
        x, it, nf = newton(B, dB, HB, x, tol=1e-10)
        w = np.linalg.eigvalsh(HB(x))
        out.append(dict(mu=mu, x=x.copy(), f=f(x), it=it, nf=nf, cond=w.max() / max(w.min(), 1e-300),
                        viol=max(0.0, np.max(g(x))), mult=-mu / g(x), gmax=np.max(g(x))))
    return x, out


NO = dict(h=lambda x: np.zeros(0), Jh=lambda x: np.zeros((0, 2)), hessh=lambda x: [],
          g=lambda x: np.zeros(0), Jg=lambda x: np.zeros((0, 2)), hessg=lambda x: [])
PROB1 = {**NO, **dict(f=f1, grad=grad1, hess=hess1, h=h1, Jh=Jh1, hessh=hessh1)}
PROB2 = {**NO, **dict(f=f2, grad=grad2, hess=hess2, g=g2, Jg=Jg2, hessg=hessg2)}
PROB3 = {**NO, **dict(f=f2, grad=grad2, hess=hess2, g=g3, Jg=Jg3, hessg=hessg3)}


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tablas"), exist_ok=True)
    TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tablas")
    
    # ================= 1D: x s.a. x>=1
    xs = np.linspace(-0.2, 2.6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for rho in (1, 5, 25):
        ax[0].plot(xs, xs + 0.5 * rho * np.maximum(0, 1 - xs)**2, label=fr"$\rho={rho}$  (mín. en $x={1-1/rho:.2f}$)")
    ax[0].plot(xs, xs, "k--", lw=1, label="$f(x)=x$")
    ax[0].axvline(1, color="gray", ls=":"); ax[0].set_title("Penalización exterior: $x+\\frac{\\rho}{2}\\max(0,1-x)^2$")
    ax[0].set_ylim(-0.2, 4); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_xlabel("$x$")
    xb = np.linspace(1.0001, 2.6, 400)
    for mu in (1.0, 0.3, 0.1):
        ax[1].plot(xb, xb - mu * np.log(xb - 1), label=fr"$\mu={mu}$  (mín. en $x={1+mu:.2f}$)")
    ax[1].plot(xs, xs, "k--", lw=1)
    ax[1].axvline(1, color="gray", ls=":"); ax[1].set_title("Barrera: $x-\\mu\\log(x-1)$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_xlabel("$x$")
    fig.tight_layout(); fig.savefig(figpath("fig14_penal_barrera_1d.png"), dpi=150); plt.close(fig)

    # ================= P2: exterior
    rhos = [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6]
    xe, ext = penalizacion_exterior(PROB2, [0.0, 0.0], rhos)
    print("EXTERIOR P2 (x0=(0,0))")
    print(f"{'rho':>8} {'x1':>10} {'x2':>10} {'f':>9} {'viol':>9} {'mu1':>7} {'mu2':>7} {'it':>4} {'nf':>4} {'cond':>9}")
    with open(os.path.join(TAB, "tabla_exterior.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrrr}\\toprule\n$\\rho$ & $x_1$ & $x_2$ & $f$ & viol.\\ & $\\hat\\mu_1$ & $\\hat\\mu_2$ & cond.\\\\\\midrule\n")
        for r in ext:
            print(f"{r['rho']:>8.0e} {r['x'][0]:>10.6f} {r['x'][1]:>10.6f} {r['f']:>9.6f} {r['viol']:>9.2e} {r['mu'][0]:>7.4f} {r['mu'][1]:>7.4f} {r['it']:>4} {r['nf']:>4} {r['cond']:>9.2e}")
            fh.write(f"$10^{{{int(round(np.log10(r['rho'])))}}}$ & {r['x'][0]:.5f} & {r['x'][1]:.5f} & {r['f']:.5f} & {r['viol']:.1e} & {r['mu'][0]:.3f} & {r['mu'][1]:.3f} & {r['cond']:.1e}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    print("  total evaluaciones de F (inner):", sum(r["nf"] for r in ext), " iteraciones internas:", sum(r["it"] for r in ext))

    # ================= P2: interior
    mus = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    xi, inte = barrera_interior(PROB2, [0.5, 1.0], mus)
    print("\nINTERIOR P2 (x0=(0.5,1))")
    print(f"{'mu':>8} {'x1':>10} {'x2':>10} {'f':>9} {'gmax':>10} {'mu1':>7} {'mu2':>7} {'it':>4} {'nf':>4} {'cond':>9}")
    with open(os.path.join(TAB, "tabla_interior.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrrrrr}\\toprule\n$\\mu$ & $x_1$ & $x_2$ & $f$ & $\\max g_i$ & $\\hat\\mu_1$ & $\\hat\\mu_2$ & cond.\\\\\\midrule\n")
        for r in inte:
            print(f"{r['mu']:>8.0e} {r['x'][0]:>10.6f} {r['x'][1]:>10.6f} {r['f']:>9.6f} {r['gmax']:>10.2e} {r['mult'][0]:>7.4f} {r['mult'][1]:>7.4f} {r['it']:>4} {r['nf']:>4} {r['cond']:>9.2e}")
            fh.write(f"$10^{{{int(round(np.log10(r['mu'])))}}}$ & {r['x'][0]:.5f} & {r['x'][1]:.5f} & {r['f']:.5f} & {r['gmax']:.1e} & {r['mult'][0]:.3f} & {r['mult'][1]:.3f} & {r['cond']:.1e}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")
    print("  total evaluaciones de B (inner):", sum(r["nf"] for r in inte), " iteraciones internas:", sum(r["it"] for r in inte))

    # ================= P1: exterior con igualdad
    xe1, ext1 = penalizacion_exterior(PROB1, [-0.3, 0.8], [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    print("\nEXTERIOR P1 (igualdad), x0=(-0.3,0.8)")
    for r in ext1:
        print(f"  rho={r['rho']:.0e} x={np.round(r['x'],6)} f={r['f']:.6f} h={h1(r['x'])[0]:+.2e} lam_est={r['lam'][0]:.4f} it={r['it']} cond={r['cond']:.1e}")

    # ================= scipy
    cons = [{"type": "ineq", "fun": lambda z: -g2(z), "jac": lambda z: -Jg2(z)}]
    r1 = minimize(f2, [0.5, 1.0], jac=grad2, constraints=cons, method="SLSQP", options={"ftol": 1e-12})
    r2 = minimize(f2, [0.5, 1.0], jac=grad2, hess=lambda z: hess2(z), constraints=[__import__("scipy.optimize", fromlist=["NonlinearConstraint"]).NonlinearConstraint(g2, -np.inf, 0, jac=Jg2, hess=lambda z, v: v[0] * hessg2(z)[0] + v[1] * hessg2(z)[1])], method="trust-constr")
    print(f"\nscipy SLSQP: {r1.x} it={r1.nit} | trust-constr: {np.round(r2.x,6)} it={r2.nit} nfev={r2.nfev}")

    # ================= Figuras: trayectorias
    def base(ax, xlim, ylim):
        xs = np.linspace(*xlim, 300); ys = np.linspace(*ylim, 300); X, Y = np.meshgrid(xs, ys)
        ax.contour(X, Y, (X - 2)**2 + (Y - 1)**2, levels=[0.25, 1, 2.25, 4, 6.25, 9], cmap="viridis", linewidths=0.7)
        ax.contourf(X, Y, ((X**2 - Y <= 0) & (X + Y - 2 <= 0)).astype(float), levels=[0.5, 1.5], colors=["#cde9c9"], alpha=0.7)
        xp = np.linspace(-1.9, 1.9, 200); ax.plot(xp, xp**2, "k", lw=1.2); ax.plot(xs, 2 - xs, "k--", lw=1.2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.plot(1, 1, "*", color="gold", mec="k", ms=16, zorder=6)
    # ================= P3 interior (1 restriccion activa en 2D) -> condicionamiento
    xi3, int3 = barrera_interior(PROB3, [0.3, 0.2], mus[:6])
    xe3, ext3 = penalizacion_exterior(PROB3, [0.3, 0.2], [1, 10, 1e2, 1e3, 1e4, 1e5, 1e6])
    print("\nINTERIOR P3 (disco)")
    for r in int3:
        print(f"  mu={r['mu']:.0e} x={np.round(r['x'],6)} f={r['f']:.6f} it={r['it']} cond={r['cond']:.2e} mult={r['mult'][0]:.4f} (mu*={MU3_STAR:.4f})")
    print("EXTERIOR P3 (disco)")
    for r in ext3:
        print(f"  rho={r['rho']:.0e} x={np.round(r['x'],6)} f={r['f']:.6f} it={r['it']} cond={r['cond']:.2e} mu_est={r['mu'][0]:.4f}")
    with open(os.path.join(TAB, "tabla_cond.tex"), "w") as fh:
        fh.write("\\begin{tabular}{rrrrr}\\toprule\n$\\rho$ o $1/\\mu$ & cond.\\ ext.\\ (P1) & cond.\\ int.\\ (P3) & err.\\ ext.\\ (P1) & err.\\ int.\\ (P3)\\\\\\midrule\n")
        for e1, i3 in zip(ext1[:6], int3[:6]):
            fh.write(f"$10^{{{int(round(np.log10(e1['rho'])))}}}$ & {e1['cond']:.1e} & {i3['cond']:.1e} & {np.linalg.norm(e1['x']-X1_STAR):.1e} & {np.linalg.norm(i3['x']-X3_STAR):.1e}\\\\\n")
        fh.write("\\bottomrule\\end{tabular}\n")


    # ---- trayectorias en P3: exterior (por fuera) e interior (por dentro) sobre el mismo grafico
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    xs_ = np.linspace(-0.4, 1.5, 300); ys_ = np.linspace(-0.2, 1.0, 300); X, Y = np.meshgrid(xs_, ys_)
    ax.contour(X, Y, (X - 2)**2 + (Y - 1)**2, levels=[1.2, 1.6, 2.2, 3.0, 4.0, 5.0], cmap="viridis", linewidths=0.7)
    th = np.linspace(0, np.pi, 300); ax.fill_between(np.cos(th), 0, np.sin(th), color="#cde9c9", alpha=0.8)
    th2 = np.linspace(0, 2 * np.pi, 400); ax.plot(np.cos(th2), np.sin(th2), "k", lw=1.5)
    pe3 = np.array([r["x"] for r in ext3]); pi3 = np.array([r["x"] for r in int3])
    ax.plot(pe3[:, 0], pe3[:, 1], "o-", color="crimson", ms=5, label=r"exterior $x(\rho)$, $\rho=1\ldots10^6$")
    ax.plot(pi3[:, 0], pi3[:, 1], "s-", color="tab:blue", ms=5, label=r"interior $x(\mu)$, $\mu=1\ldots10^{-5}$")
    for r in ext3[:2]:
        ax.annotate(f"$\\rho=10^{{{int(round(np.log10(r['rho'])))}}}$", r["x"], textcoords="offset points", xytext=(5, 6), fontsize=8, color="crimson")
    for r in int3[:2]:
        ax.annotate(f"$\\mu=10^{{{int(round(np.log10(r['mu'])))}}}$", r["x"], textcoords="offset points", xytext=(-38, -12), fontsize=8, color="tab:blue")
    ax.plot(*X3_STAR, "*", color="gold", mec="k", ms=16, zorder=6, label=r"$x^\ast$")
    ax.text(0.15, 0.35, "región\nfactible", fontsize=10, ha="center")
    ax.set_xlim(-0.4, 1.5); ax.set_ylim(-0.2, 1.0); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="upper left"); ax.set_title("P3: el exterior llega por fuera, el interior por dentro")
    fig.tight_layout(); fig.savefig(figpath("fig14_trayectorias_P3.png"), dpi=150); plt.close(fig)

    # ---- zoom en P2
    fig, ax = plt.subplots(figsize=(6.6, 5.0)); base(ax, (0.35, 1.35), (0.8, 1.15))
    pe = np.array([r["x"] for r in ext]); pi = np.array([[0.5, 1.0]] + [r["x"] for r in inte])
    ax.plot(pe[:, 0], pe[:, 1], "o-", color="crimson", ms=5, label="exterior (por fuera)")
    ax.plot(pi[:, 0], pi[:, 1], "s-", color="tab:blue", ms=5, label="interior (por dentro)")
    ax.plot(0.5, 1.0, "k^", ms=8, label="inicio interior")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("P2 (zoom): ambas trayectorias terminan en $x^\\ast=(1,1)$")
    fig.tight_layout(); fig.savefig(figpath("fig14_trayectorias_P2_zoom.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].loglog([r["rho"] for r in ext1], [np.linalg.norm(r["x"] - X1_STAR) for r in ext1], "o-", color="crimson", label=r"exterior (P1) vs $\rho$")
    ax[0].loglog([1 / r["mu"] for r in int3], [np.linalg.norm(r["x"] - X3_STAR) for r in int3], "s-", color="tab:blue", label=r"interior (P3) vs $1/\mu$")
    ax[0].set_xlabel(r"$\rho$   (o   $1/\mu$)"); ax[0].set_ylabel(r"$\|x-x^\ast\|$"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
    ax[0].set_title("Error: baja como $1/\\rho$  ($\\mu$)")
    ax[1].loglog([r["rho"] for r in ext1], [r["cond"] for r in ext1], "o-", color="crimson", label="exterior (P1)")
    ax[1].loglog([1 / r["mu"] for r in int3], [r["cond"] for r in int3], "s-", color="tab:blue", label="interior (P3)")
    ax[1].set_xlabel(r"$\rho$   (o   $1/\mu$)"); ax[1].set_ylabel(r"cond$(\nabla^2)$ del subproblema"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    ax[1].set_title("Mal condicionamiento: crece como $\\rho$  ($1/\\mu$)")
    fig.tight_layout(); fig.savefig(figpath("fig14_error_cond.png"), dpi=150); plt.close(fig)
