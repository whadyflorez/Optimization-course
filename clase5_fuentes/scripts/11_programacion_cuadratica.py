"""
Programacion cuadratica (QP):   min 0.5 x'Hx + c'x   s.a.  A x = b,   G x <= h.

1) QP con igualdades: se resuelve UN sistema lineal (el sistema KKT).
2) QP con desigualdades: metodo de CONJUNTO ACTIVO (primal) a mano, con el
   ejemplo 16.4 de Nocedal & Wright, y comparacion con scipy.
"""
import os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath


# ---------------------------------------------------------------- igualdades
def qp_igualdad(H, c, A, b):
    """min 0.5 x'Hx + c'x  s.a. Ax=b  -> resuelve [[H, A'],[A, 0]] [x; lam] = [-c; b]."""
    n, m = H.shape[0], A.shape[0]
    K = np.block([[H, A.T], [A, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.concatenate([-c, b]))
    return sol[:n], sol[n:]


# ---------------------------------------------------------- conjunto activo
def qp_conjunto_activo(H, c, G, h, x0, W0, tol=1e-10, max_iter=50, verbose=True):
    """Metodo de conjunto activo primal (convexo, x0 factible, W0 = restricciones
    activas en x0 linealmente independientes).  Restricciones: G x <= h."""
    x = np.array(x0, float)
    W = list(W0)
    hist = [(x.copy(), list(W), "inicio")]
    for k in range(max_iter):
        g = H @ x + c
        # --- subproblema: min 0.5 p'Hp + g'p  s.a.  G_W p = 0
        if W:
            A = G[W]
            p, _ = qp_igualdad(H, g, A, np.zeros(len(W)))
        else:
            p = np.linalg.solve(H, -g)
        if np.linalg.norm(p) < tol:
            # x es el minimizador sobre W: revisar multiplicadores
            if not W:
                hist.append((x.copy(), list(W), "optimo"))
                break
            mu = np.linalg.solve(G[W] @ G[W].T, -G[W] @ g)    # g + G_W' mu = 0
            if verbose:
                print(f"  it {k}: x={np.round(x,4)} W={[i+1 for i in W]}  p=0  mu={np.round(mu,3)}")
            if np.all(mu >= -tol):
                hist.append((x.copy(), list(W), "optimo"))
                break
            j = W[int(np.argmin(mu))]
            W.remove(j)                                       # soltar la restriccion con mu<0 mas negativa
            hist.append((x.copy(), list(W), f"suelta c{j+1}"))
        else:
            # paso maximo antes de chocar con una restriccion fuera de W
            alpha, blk = 1.0, None
            for i in range(G.shape[0]):
                if i in W: continue
                Gp = G[i] @ p
                if Gp > 1e-12:
                    a_i = (h[i] - G[i] @ x) / Gp
                    if a_i < alpha:
                        alpha, blk = a_i, i
            if verbose:
                print(f"  it {k}: x={np.round(x,4)} W={[i+1 for i in W]}  p={np.round(p,3)}  alpha={alpha:.3f}"
                      + (f"  (choca con c{blk+1})" if blk is not None else ""))
            x = x + alpha * p
            if blk is not None:
                W.append(blk)
            hist.append((x.copy(), list(W), "paso" + (f", agrega c{blk+1}" if blk is not None else "")))
    return x, W, hist


if __name__ == "__main__":
    # ================= 1) QP con igualdad:  min x'x  s.a. x1+2x2+3x3 = 6
    H = 2 * np.eye(3); c = np.zeros(3); A = np.array([[1., 2., 3.]]); b = np.array([6.])
    x, lam = qp_igualdad(H, c, A, b)
    print("QP igualdad:", x, "lambda =", lam, " (teoria x = 6/14 *(1,2,3) =", 6 / 14 * np.array([1, 2, 3]), ")")
    res = minimize(lambda z: z @ z, np.zeros(3), constraints=[{"type": "eq", "fun": lambda z: A @ z - b}], method="SLSQP")
    print("  scipy SLSQP:", res.x)

    # ================= 2) QP con desigualdades (N&W 16.4):  min (x1-1)^2 + (x2-2.5)^2
    H = 2 * np.eye(2); c = np.array([-2., -5.])
    G = np.array([[-1.,  2.],     # c1:  x1 - 2 x2 + 2 >= 0   ->  -x1 + 2 x2 <= 2
                  [ 1.,  2.],     # c2: -x1 - 2 x2 + 6 >= 0   ->   x1 + 2 x2 <= 6
                  [ 1., -2.],     # c3: -x1 + 2 x2 + 2 >= 0   ->   x1 - 2 x2 <= 2
                  [-1.,  0.],     # c4:  x1 >= 0
                  [ 0., -1.]])    # c5:  x2 >= 0
    h = np.array([2., 6., 2., 0., 0.])
    x0 = np.array([2., 0.])
    print("\nConjunto activo (x0 = (2,0), W0 = {c3, c5}):")
    xs, Ws, hist = qp_conjunto_activo(H, c, G, h, x0, [2, 4])
    fval = 0.5 * xs @ H @ xs + c @ xs + 1 + 6.25
    print("  solucion:", xs, " W final =", [i + 1 for i in Ws], " f =", fval)
    cons = [{"type": "ineq", "fun": lambda z: h - G @ z, "jac": lambda z: -G}]
    ref = minimize(lambda z: 0.5 * z @ H @ z + c @ z, x0, jac=lambda z: H @ z + c, constraints=cons, method="SLSQP", options={"ftol": 1e-14})
    ref2 = minimize(lambda z: 0.5 * z @ H @ z + c @ z, x0, jac=lambda z: H @ z + c, hess=lambda z: H, constraints=cons, method="trust-constr")
    print("  scipy SLSQP:", ref.x, "iters", ref.nit, "| trust-constr:", np.round(ref2.x, 6), "iters", ref2.nit)

    # enumeracion: 2^5 conjuntos activos posibles
    print("\nEnumeracion de conjuntos activos (KKT por casos), m=5 -> 2^5=32 casos")
    cuenta = 0
    for r in range(0, 4):
        for S in itertools.combinations(range(5), r):
            S = list(S)
            if S:
                K = np.block([[H, G[S].T], [G[S], np.zeros((len(S), len(S)))]])
                try:
                    sol = np.linalg.solve(K, np.concatenate([-c, h[S]]))
                except np.linalg.LinAlgError:
                    continue
                x_, mu_ = sol[:2], sol[2:]
            else:
                x_, mu_ = np.linalg.solve(H, -c), np.array([])
            if np.all(G @ x_ - h <= 1e-9) and np.all(mu_ >= -1e-9):
                cuenta += 1
                print("  KKT cumplido con S =", [i + 1 for i in S], " x =", np.round(x_, 4), " mu =", np.round(mu_, 4))
    print("  puntos KKT encontrados:", cuenta)

    # ---------------- figura: region, curvas de nivel y trayectoria
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    xx = np.linspace(-0.3, 4.2, 300); yy = np.linspace(-0.3, 3.4, 300)
    X, Y = np.meshgrid(xx, yy)
    feas = np.ones_like(X, bool)
    for gi, hi in zip(G, h):
        feas &= gi[0] * X + gi[1] * Y <= hi + 1e-12
    ax.contourf(X, Y, feas.astype(float), levels=[0.5, 1.5], colors=["#cde9c9"], alpha=0.9)
    ax.contour(X, Y, (X - 1)**2 + (Y - 2.5)**2, levels=[0.2, 0.8, 1.5, 2.5, 4, 6], cmap="viridis", linewidths=0.8)
    xl = np.linspace(-0.3, 4.2, 10)
    ax.plot(xl, (xl + 2) / 2, "k", lw=1, label="c1"); ax.plot(xl, (6 - xl) / 2, "k", lw=1)
    ax.plot(xl, (xl - 2) / 2, "k", lw=1)
    pts = np.array([hh[0] for hh in hist])
    ax.plot(pts[:, 0], pts[:, 1], "o-", color="crimson", lw=1.6, ms=6, label="iterados (conjunto activo)")
    for kk, hh in enumerate(hist):
        pass
    ax.plot(*x0, "s", color="black", ms=8, label="inicio")
    ax.plot(*xs, "*", color="gold", mec="k", ms=18, zorder=6, label=r"$x^\ast=(1.4,1.7)$")
    ax.text(3.0, 0.45, "c3", fontsize=10); ax.text(0.15, 1.35, "c1", fontsize=10); ax.text(3.3, 1.5, "c2", fontsize=10)
    ax.text(0.05, 0.12, "c5: $x_2\\geq0$", fontsize=8); ax.text(0.03, 0.6, "c4:\n$x_1\\geq0$", fontsize=8)
    for kk in range(len(pts)):
        pass
    etiquetas = [("$x_0$", (2.0, 0.0), (6, -14)), ("$x_2$", (1.0, 0.0), (-4, -14)), ("$x_3$", (1.0, 1.5), (-20, 0)), ("$x_4=x^\\ast$", (1.4, 1.7), (6, 6))]
    for txt, pt, off in etiquetas:
        ax.annotate(txt, pt, textcoords="offset points", xytext=off, fontsize=9)
    ax.set_xlim(-0.3, 4.2); ax.set_ylim(-0.3, 3.4); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("QP: método de conjunto activo")
    fig.tight_layout(); fig.savefig(figpath("fig11_qp_conjunto_activo.png"), dpi=150); plt.close(fig)

    # ---------------- figura: QP con igualdad -> geometria (min x'Hx sobre una recta, 2D)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    xx = np.linspace(-1, 4, 300); X, Y = np.meshgrid(xx, xx)
    ax.contour(X, Y, (X - 0.5)**2 + 2 * (Y - 0.5)**2, levels=8, cmap="viridis", linewidths=0.8)
    ax.plot(xx, 3 - xx, "k", lw=2, label=r"$x_1+x_2=3$")
    Hq = np.diag([2., 4.]); cq = np.array([-1., -2.]); Aq = np.array([[1., 1.]]); bq = np.array([3.])
    xq, lq = qp_igualdad(Hq, cq, Aq, bq)
    ax.plot(*xq, "*", color="gold", mec="k", ms=18, label=f"$x^*=({xq[0]:.3f},{xq[1]:.3f})$")
    ax.plot(0.5, 0.5, "s", color="gray", label="mín. libre")
    ax.set_xlim(-1, 4); ax.set_ylim(-1, 4); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    fig.tight_layout(); fig.savefig(figpath("fig11_qp_igualdad.png"), dpi=150); plt.close(fig)
    print("\nQP igualdad 2D:", xq, lq)
