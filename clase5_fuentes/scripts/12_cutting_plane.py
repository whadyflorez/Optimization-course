"""
Cutting plane (metodo de Kelley) para  min f(x) s.a. g(x) <= 0  (f, g convexas).

IDEA: reemplazar f y g por sus LINEALIZACIONES (cortes) y resolver un LP.
  Forma epigrafo:    min t   s.a.  f(x_k) + grad f(x_k)'(x - x_k) <= t        (corte de objetivo)
                                   g_i(x_k) + grad g_i(x_k)'(x - x_k) <= 0   (corte de restriccion)
                                   x en una caja [-B,B]^n  (para que el LP este acotado)
En cada iteracion se resuelve el LP con TODOS los cortes acumulados y se agrega
un corte nuevo en la solucion encontrada.  Como f, g convexas: la linealizacion
queda DEBAJO de la funcion -> el LP es una relajacion -> t_k es COTA INFERIOR de f*.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _problemas import *


def cutting_plane(f, grad, g, Jg, x0, B=4.0, tol=1e-6, max_iter=60):
    n = len(x0)
    x = np.array(x0, float)
    A_ub, b_ub = [], []          # cortes acumulados en las variables (x, t)
    hist = []
    for k in range(max_iter):
        # --- agregar cortes en el punto actual x_k
        fk, gk = f(x), grad(x)
        A_ub.append(np.concatenate([gk, [-1.0]]))                 # gk'(x) - t <= gk'xk - fk
        b_ub.append(gk @ x - fk)
        gv, Jv = g(x), Jg(x)
        for i in range(len(gv)):
            A_ub.append(np.concatenate([Jv[i], [0.0]]))           # Jv_i'(x) <= Jv_i'xk - g_i(xk)
            b_ub.append(Jv[i] @ x - gv[i])
        # --- resolver el LP:  min t
        cost = np.zeros(n + 1); cost[-1] = 1.0
        res = linprog(cost, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=[(-B, B)] * n + [(None, None)], method="highs")
        x_new, t_k = res.x[:n], res.x[-1]
        viol = max(0.0, np.max(g(x_new)))
        hist.append(dict(k=k, x=x_new.copy(), t=t_k, f=f(x_new), viol=viol, ncuts=len(A_ub)))
        if viol < tol and f(x_new) - t_k < tol:                   # factible y brecha cerrada
            x = x_new
            break
        x = x_new
    return x, hist, (np.array(A_ub), np.array(b_ub))


def figura(nombre, titulo, Acut, bcut, m, hist, x0, Kc, f_nivel, g_contornos, xlim, ylim, xstar, feas_fn, curvas):
    """Dibuja iterados y la region del LP formada por los cortes de RESTRICCION de las primeras Kc iteraciones."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs = np.linspace(*xlim, 300); ys = np.linspace(*ylim, 300)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, f_nivel(X, Y), levels=[0.25, 1, 2.25, 4, 6.25, 9], cmap="viridis", linewidths=0.7)
    ax.contourf(X, Y, feas_fn(X, Y).astype(float), levels=[0.5, 1.5], colors=["#cde9c9"], alpha=0.7)
    lp = np.ones_like(X, bool)
    r = 1 + m
    for k in range(Kc):
        for j in range(1, r):
            row = Acut[r * k + j]; rhs = bcut[r * k + j]
            lp &= row[0] * X + row[1] * Y <= rhs + 1e-12
    ax.contour(X, Y, lp.astype(float), levels=[0.5], colors=["crimson"], linewidths=1.6)
    for c in curvas:
        ax.plot(*c[:2], *c[2:])
    pts = np.array([x0] + [h_["x"] for h_ in hist[:Kc]])
    ax.plot(pts[:, 0], pts[:, 1], "o-", color="darkorange", ms=5, lw=1.2, label="iterados del LP")
    ax.plot(*x0, "s", color="black", ms=8, label="inicio")
    ax.plot(*xstar, "*", color="gold", mec="k", ms=18, zorder=6, label=r"$x^\ast$")
    ax.plot([], [], color="crimson", lw=1.6, label="región del LP (cortes)")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(titulo)
    fig.tight_layout(); fig.savefig(figpath(nombre), dpi=150); plt.close(fig)


if __name__ == "__main__":
    fN = lambda X, Y: (X - 2)**2 + (Y - 1)**2
    resumen = {}
    # ====================== P2 (optimo = vertice)
    x0 = np.array([0.0, 0.0])
    x, hist, (A2, b2) = cutting_plane(f2, grad2, g2, Jg2, x0, tol=1e-6)
    print(f"[P2] Cutting plane: {len(hist)} iteraciones, x={x}, f={f2(x):.8f}")
    for h_ in hist:
        print(f"  k={h_['k']} x=({h_['x'][0]:.5f},{h_['x'][1]:.5f}) t_k={h_['t']:.6f} f={h_['f']:.6f} viol={h_['viol']:.2e} cortes={h_['ncuts']}")
    resumen["P2"] = len(hist)
    xp = np.linspace(-1.9, 1.9, 200)
    figura("fig12_cp_P2.png", f"P2: cutting plane, {len(hist)} iteraciones", A2, b2, 2, hist, x0, len(hist), fN, None,
           (-2.6, 3.0), (-0.8, 4.6), X2_STAR, lambda X, Y: (X**2 - Y <= 0) & (X + Y - 2 <= 0),
           [(xp, xp**2, "k-", 1.0), (np.linspace(-2.6, 3, 10), 2 - np.linspace(-2.6, 3, 10), "k--")]
           if False else [])
    # ====================== P3 (optimo en frontera curva)
    x0 = np.array([0.0, 0.0])
    x, hist3, (A3, b3) = cutting_plane(f2, grad2, g3, Jg3, x0, tol=1e-6, max_iter=200)
    print(f"\n[P3] Cutting plane: {len(hist3)} iteraciones, x={x}, f={f2(x):.8f} (f*={F3_STAR:.8f}), x*={X3_STAR}")
    for h_ in hist3[:10] + hist3[-3:]:
        print(f"  k={h_['k']} x=({h_['x'][0]:.5f},{h_['x'][1]:.5f}) t_k={h_['t']:.6f} f={h_['f']:.6f} viol={h_['viol']:.2e} cortes={h_['ncuts']}")
    resumen["P3"] = len(hist3)
    ref = minimize(f2, [0., 0.], jac=grad2, method="SLSQP", options={"ftol": 1e-14},
                   constraints=[{"type": "ineq", "fun": lambda z: -g3(z), "jac": lambda z: -Jg3(z)}])
    print("scipy SLSQP (P3):", ref.x, "iters", ref.nit)
    th = np.linspace(0, 2 * np.pi, 400)
    circ = [(np.cos(th), np.sin(th), "k-")]
    for Kc, nm in ((3, "k3"), (6, "k6"), (12, "k12"), (len(hist3), "final")):
        figura(f"fig12_cp_P3_{nm}.png", f"P3: cutting plane, {min(Kc, len(hist3))} iteraciones", A3, b3, 1, hist3, x0,
               min(Kc, len(hist3)), fN, None, (-1.6, 2.6), (-1.6, 2.2), X3_STAR,
               lambda X, Y: X**2 + Y**2 <= 1, circ)

    # ---------- convergencia
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ks = [h_["k"] + 1 for h_ in hist3]
    ax.semilogy(ks, [max(F3_STAR - h_["t"], 1e-12) for h_ in hist3], "o-", ms=3, label=r"$f^\ast - t_k$ (cota inferior)")
    ax.semilogy(ks, [max(h_["viol"], 1e-12) for h_ in hist3], "s-", ms=3, label="violación de restricción")
    ks2 = [h_["k"] + 1 for h_ in hist]
    ax.semilogy(ks2, [max(F2_STAR - h_["t"], 1e-12) for h_ in hist], "^--", color="gray", label="P2: $f^\\ast - t_k$")
    ax.set_xlabel("iteración"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("Cutting plane: convergencia (P3 y P2)")
    fig.tight_layout(); fig.savefig(figpath("fig12_cp_conv.png"), dpi=150); plt.close(fig)

    # ---------- 1D
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    xx = np.linspace(-0.2, 3.2, 300); fx = lambda z: (z - 2.2)**2
    ax.plot(xx, fx(xx), "k", lw=2, label=r"$f(x)=(x-2.2)^2$")
    P = [0.0, 3.0, 1.1, 2.0]
    cortes = [(fx(p) + 2 * (p - 2.2) * (xx - p)) for p in P]
    for c in cortes:
        ax.plot(xx, c, "--", lw=0.9, color="tab:blue", alpha=0.6)
    ax.plot(xx, np.max(cortes, axis=0), color="crimson", lw=2.2, label="máximo de los cortes (modelo)")
    ax.plot(P, [fx(p) for p in P], "o", color="darkorange", label="puntos de corte")
    ax.set_ylim(-1, 6); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_xlabel("$x$")
    ax.set_title("Cutting plane en 1D: el modelo poliédrico queda por debajo de $f$")
    fig.tight_layout(); fig.savefig(figpath("fig12_cp_1d.png"), dpi=150); plt.close(fig)
