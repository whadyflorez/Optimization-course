"""
Programacion cuadratica secuencial (SQP), version didactica.

En cada iteracion se resuelve el SUBPROBLEMA QP  (modelo cuadratico del Lagrangiano +
restricciones LINEALIZADAS):
        min_p   grad f(x_k)' p + 0.5 p' W_k p
        s.a.    h(x_k) + Jh(x_k) p = 0
                g(x_k) + Jg(x_k) p <= 0
y se actualiza  x_{k+1} = x_k + alpha_k p_k,  con alpha_k de una busqueda lineal sobre la
funcion de merito l1:   phi(x) = f(x) + nu * ( sum|h_i| + sum max(0, g_j) ).
W_k = Hessiana del Lagrangiano (exacta) o una aproximacion BFGS con amortiguamiento de Powell.
El QP se resuelve por enumeracion de conjuntos activos (exacto y facil de entender
cuando hay pocas desigualdades).
"""
import os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _problemas import *
from _functions import rosen_f, rosen_grad, rosen_hess


def resolver_qp(W, gf, Aeq, ceq, Ain, cin, tol=1e-9):
    """min gf'p + 0.5 p'Wp  s.a. Aeq p + ceq = 0,  Ain p + cin <= 0.
    Devuelve (p, lam, mu, activas). Recorre subconjuntos S de desigualdades activas:
    resuelve el sistema KKT de igualdad y acepta el que cumple factibilidad y mu>=0."""
    n, me, mi = len(gf), len(ceq), len(cin)
    for r in range(mi + 1):
        for S in itertools.combinations(range(mi), r):
            S = list(S)
            A = np.vstack([Aeq.reshape(me, n), Ain[S].reshape(len(S), n)]) if (me + len(S)) else np.zeros((0, n))
            rhs = np.concatenate([-ceq, -cin[S]]) if (me + len(S)) else np.zeros(0)
            K = np.block([[W, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
            try:
                sol = np.linalg.solve(K, np.concatenate([-gf, rhs]))
            except np.linalg.LinAlgError:
                continue
            p, nu = sol[:n], sol[n:]
            lam, mu_S = nu[:me], nu[me:]
            if mi and np.any(Ain @ p + cin > tol):
                continue
            if len(S) and np.any(mu_S < -tol):
                continue
            mu = np.zeros(mi); mu[S] = mu_S
            return p, lam, mu, S
    raise RuntimeError("QP sin solucion KKT (linealizacion inconsistente)")


def sqp(prob, x0, modo="exacta", tol=1e-9, max_iter=50, verbose=False):
    f, grad, hessL = prob["f"], prob["grad"], prob["hessL"]
    h, Jh, g, Jg = prob["h"], prob["Jh"], prob["g"], prob["Jg"]
    x = np.array(x0, float); n = len(x)
    me, mi = len(h(x)), len(g(x))
    # multiplicadores iniciales: minimos cuadrados sobre las restricciones de igualdad
    lam = np.linalg.lstsq(Jh(x).T, -grad(x), rcond=None)[0] if me else np.zeros(0)
    mu = np.zeros(mi)
    B = np.eye(n)
    nu_pen = 1.0
    nf = 0
    def viol(z):
        return (np.sum(np.abs(h(z))) if me else 0.0) + (np.sum(np.maximum(0, g(z))) if mi else 0.0)
    def phi(z, nu):
        return f(z) + nu * viol(z)
    def gradL(z, l, m_):
        v = grad(z).copy()
        if me: v += Jh(z).T @ l
        if mi: v += Jg(z).T @ m_
        return v
    hist = []
    for k in range(max_iter):
        r_est = np.max(np.abs(gradL(x, lam, mu)))
        r_comp = np.max(np.abs(mu * g(x))) if mi else 0.0
        kkt = max(r_est, viol(x), r_comp)
        hist.append(dict(k=k, x=x.copy(), f=f(x), viol=viol(x), kkt=kkt, alpha=np.nan, act=None))
        if verbose:
            print(f"  k={k} x={np.round(x,6)} f={f(x):.8f} viol={viol(x):.2e} kkt={kkt:.2e}")
        if kkt < tol:
            break
        W = hessL(x, lam, mu) if modo == "exacta" else B
        w_min = np.linalg.eigvalsh(W).min()
        if w_min < 1e-6:
            W = W + (1e-6 - w_min + 1e-3) * np.eye(n)          # salvaguarda: W definida positiva
        gf = grad(x)
        Aeq = Jh(x) if me else np.zeros((0, n)); ceq = h(x) if me else np.zeros(0)
        Ain = Jg(x) if mi else np.zeros((0, n)); cin = g(x) if mi else np.zeros(0)
        p, lam_new, mu_new, S = resolver_qp(W, gf, Aeq, ceq, Ain, cin)
        # --- penalizacion de la funcion de merito y busqueda lineal (Armijo)
        mult_max = max(np.max(np.abs(np.concatenate([lam_new, mu_new]))) if (me + mi) else 0.0, 0.0)
        nu_pen = max(nu_pen, 1.1 * mult_max + 1e-3)
        D = gf @ p - nu_pen * viol(x)
        alpha = 1.0
        while phi(x + alpha * p, nu_pen) > phi(x, nu_pen) + 1e-4 * alpha * D and alpha > 1e-8:
            alpha *= 0.5; nf += 1
        nf += 1
        x_new = x + alpha * p
        lam_next = lam + alpha * (lam_new - lam); mu_next = mu + alpha * (mu_new - mu)
        if modo == "bfgs":
            s = x_new - x
            y = gradL(x_new, lam_next, mu_next) - gradL(x, lam_next, mu_next)
            sBs = s @ B @ s
            theta = 1.0 if s @ y >= 0.2 * sBs else 0.8 * sBs / (sBs - s @ y)   # amortiguamiento de Powell
            r_ = theta * y + (1 - theta) * (B @ s)
            if s @ r_ > 1e-14:
                B = B - np.outer(B @ s, B @ s) / sBs + np.outer(r_, r_) / (s @ r_)
        hist[-1].update(alpha=alpha, act=[i + 1 for i in S])
        x, lam, mu = x_new, lam_next, mu_next
    return x, lam, mu, hist, nf


PROB1 = dict(f=f1, grad=grad1, hessL=lambda x, l, m: hess1(x) + l[0] * hessh1(x)[0],
             h=h1, Jh=Jh1, g=lambda x: np.zeros(0), Jg=lambda x: np.zeros((0, 2)))
PROB2 = dict(f=f2, grad=grad2, hessL=lambda x, l, m: hess2(x) + m[0] * hessg2(x)[0] + m[1] * hessg2(x)[1],
             h=lambda x: np.zeros(0), Jh=lambda x: np.zeros((0, 2)), g=g2, Jg=Jg2)

NOH = dict(h=lambda x: np.zeros(0), Jh=lambda x: np.zeros((0, 2)))
PROB3 = dict(f=f2, grad=grad2, hessL=lambda x, l, m: hess2(x) + m[0] * hessg3(x)[0], g=g3, Jg=Jg3, **NOH)
PROB4 = dict(f=rosen_f, grad=rosen_grad, hessL=lambda x, l, m: rosen_hess(x) + m[0] * hessg3(x)[0], g=g3, Jg=Jg3, **NOH)

if __name__ == "__main__":
    # ============ P1 (igualdad) ============
    print("=== P1: min x1+x2 s.a. x1^2+x2^2=2 ===")
    for x0 in ([-2.0, -0.5], [-1.5, 0.5], [0.5, -2.0], [1.5, 1.0]):
        for modo in ("exacta", "bfgs"):
            x, lam, mu, hist, nf = sqp(PROB1, x0, modo)
            print(f"  x0={x0} {modo:6s}: {len(hist)-1:2d} it  x={np.round(x,8)}  lam={np.round(lam,6)}  f={f1(x):.8f}")
    print("\n-- traza P1 (Hessiana exacta, x0=(-2,-0.5)):")
    x, lam, mu, hist, nf = sqp(PROB1, [-2.0, -0.5], "exacta", verbose=True)
    print("-- traza P1 (BFGS):")
    x, lam, mu, hist, nf = sqp(PROB1, [-2.0, -0.5], "bfgs", verbose=True)

    # ============ P2 (desigualdades) ============
    print("\n=== P2 ===")
    for x0 in ([0.0, 0.0], [0.0, 2.0], [-1.0, 3.0], [2.0, 3.0], [1.5, 0.0]):
        for modo in ("exacta", "bfgs"):
            x, lam, mu, hist, nf = sqp(PROB2, x0, modo)
            print(f"  x0={x0} {modo:6s}: {len(hist)-1:2d} it  x={np.round(x,8)}  mu={np.round(mu,6)}  f={f2(x):.8f}")
    print("\n-- traza P2 (exacta, x0=(0,0)):")
    xe, lam, mu, hist_e, nf = sqp(PROB2, [0.0, 0.0], "exacta", verbose=True)
    print("   QP activas por iteracion:", [h_["act"] for h_ in hist_e], " alpha:", [round(h_["alpha"], 3) for h_ in hist_e[:-1]])
    print("-- traza P2 (BFGS, x0=(0,0)):")
    xb, lam, mu, hist_b, nf = sqp(PROB2, [0.0, 0.0], "bfgs", verbose=True)
    ref = minimize(f2, [0., 0.], jac=grad2, method="SLSQP", options={"ftol": 1e-14},
                   constraints=[{"type": "ineq", "fun": lambda z: -g2(z), "jac": lambda z: -Jg2(z)}])
    print("scipy SLSQP:", ref.x, "iters", ref.nit, "nfev", ref.nfev)

    # ---------- figuras: trayectorias P2 y P1
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs = np.linspace(-1.2, 3.0, 300); ys = np.linspace(-0.6, 3.8, 300)
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, (X - 2)**2 + (Y - 1)**2, levels=[0.25, 1, 2.25, 4, 6.25, 9], cmap="viridis", linewidths=0.7)
    ax.contourf(X, Y, ((X**2 - Y <= 0) & (X + Y - 2 <= 0)).astype(float), levels=[0.5, 1.5], colors=["#cde9c9"], alpha=0.7)
    xp = np.linspace(-1.2, 1.9, 200); ax.plot(xp, xp**2, "k", lw=1.2); ax.plot(xs, 2 - xs, "k--", lw=1.2)
    pe = np.array([h_["x"] for h_ in hist_e]); pb = np.array([h_["x"] for h_ in hist_b])
    ax.plot(pe[:, 0], pe[:, 1], "o-", color="crimson", ms=6, lw=1.4, label=f"SQP Hessiana exacta ({len(hist_e)-1} it.)")
    ax.plot(pb[:, 0], pb[:, 1], "s--", color="tab:blue", ms=5, lw=1.2, label=f"SQP BFGS ({len(hist_b)-1} it.)")
    ax.plot(0, 0, "k^", ms=8, label="inicio"); ax.plot(1, 1, "*", color="gold", mec="k", ms=18, zorder=6, label=r"$x^\ast$")
    ax.set_xlim(-1.2, 3.0); ax.set_ylim(-0.6, 3.8); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(fontsize=7.5, loc="upper right"); ax.set_title("SQP en P2")
    fig.tight_layout(); fig.savefig(figpath("fig13_sqp_P2.png"), dpi=150); plt.close(fig)

    x1e, _, _, h1e, _ = sqp(PROB1, [-2.0, -0.5], "exacta")
    x1b, _, _, h1b, _ = sqp(PROB1, [-2.0, -0.5], "bfgs")
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    xs = np.linspace(-2.6, 2.0, 200); X, Y = np.meshgrid(xs, xs)
    ax.contour(X, Y, X + Y, levels=np.arange(-4, 4.1, 0.5), cmap="coolwarm", linewidths=0.7)
    th = np.linspace(0, 2 * np.pi, 300); ax.plot(np.sqrt(2) * np.cos(th), np.sqrt(2) * np.sin(th), "k", lw=2, label=r"$h(x)=0$")
    pe = np.array([h_["x"] for h_ in h1e]); pb = np.array([h_["x"] for h_ in h1b])
    ax.plot(pe[:, 0], pe[:, 1], "o-", color="crimson", ms=6, label=f"SQP exacta ({len(h1e)-1} it.)")
    ax.plot(pb[:, 0], pb[:, 1], "s--", color="tab:blue", ms=5, label=f"SQP BFGS ({len(h1b)-1} it.)")
    ax.plot(-2, -0.5, "k^", ms=8, label="inicio"); ax.plot(-1, -1, "*", color="gold", mec="k", ms=18, zorder=6)
    ax.set_xlim(-2.6, 2.0); ax.set_ylim(-2.6, 2.0); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title("SQP en P1")
    fig.tight_layout(); fig.savefig(figpath("fig13_sqp_P1.png"), dpi=150); plt.close(fig)

    # ---------- convergencia: error vs iteracion (P1 y P3)
    x3e, _, mu3, h3e, _ = sqp(PROB3, [2.0, 2.0], "exacta")
    x3b, _, mu3b, h3b, _ = sqp(PROB3, [2.0, 2.0], "bfgs")
    print(f"\n=== P3 (disco) x0=(2,2): exacta {len(h3e)-1} it x={x3e} mu={mu3} | BFGS {len(h3b)-1} it x={x3b} mu={mu3b}")
    print("   errores exacta:", [f"{np.linalg.norm(h_['x']-X3_STAR):.1e}" for h_ in h3e])
    print("   errores BFGS  :", [f"{np.linalg.norm(h_['x']-X3_STAR):.1e}" for h_ in h3b])
    print("   errores P1 exacta:", [f"{np.linalg.norm(h_['x']-X1_STAR):.1e}" for h_ in h1e])
    print("   errores P1 BFGS  :", [f"{np.linalg.norm(h_['x']-X1_STAR):.1e}" for h_ in h1b])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for a, (He, Hb, xs_, ttl) in zip(ax, ((h1e, h1b, X1_STAR, "P1 (igualdad)"), (h3e, h3b, X3_STAR, "P3 (disco)"))):
        a.semilogy([max(np.linalg.norm(h_["x"] - xs_), 1e-16) for h_ in He], "o-", label="SQP Hessiana exacta")
        a.semilogy([max(np.linalg.norm(h_["x"] - xs_), 1e-16) for h_ in Hb], "s--", label="SQP BFGS")
        a.set_xlabel("iteración"); a.set_title(ttl); a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    ax[0].set_ylabel(r"$\|x_k-x^\ast\|$")
    fig.tight_layout(); fig.savefig(figpath("fig13_sqp_conv.png"), dpi=150); plt.close(fig)

    # ---------- P4: Rosenbrock en el disco unidad (no convexo)
    for x0 in ([0.0, 0.5], [-0.5, 0.0], [0.5, -0.5]):
        for modo in ("exacta", "bfgs"):
            x4, _, mu4, h4, _ = sqp(PROB4, x0, modo, max_iter=100)
            print(f"P4 x0={x0} {modo:6s}: {len(h4)-1:3d} it x={np.round(x4,6)} f={rosen_f(x4):.6f} mu={np.round(mu4,4)} kkt={h4[-1]['kkt']:.1e}")
    ref4 = minimize(rosen_f, [0.0, 0.5], jac=rosen_grad, method="SLSQP", options={"ftol": 1e-14},
                    constraints=[{"type": "ineq", "fun": lambda z: -g3(z), "jac": lambda z: -Jg3(z)}])
    print("scipy SLSQP P4:", ref4.x, ref4.fun, "it", ref4.nit)
