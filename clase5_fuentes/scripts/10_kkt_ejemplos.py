"""
KKT: geometria, verificacion numerica, enumeracion de conjuntos activos y
sensibilidad de los multiplicadores (precio sombra).
"""
import os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _problemas import *


def residuos_kkt(x, mu, grad, g, Jg):
    """Devuelve (estacionariedad, factibilidad primal, factibilidad dual, complementariedad)."""
    est = np.linalg.norm(grad(x) + Jg(x).T @ mu)
    fac = max(0.0, np.max(g(x)))
    dual = max(0.0, -np.min(mu))
    comp = np.max(np.abs(mu * g(x)))
    return est, fac, dual, comp


def multiplicadores_activos(x, grad, g, Jg, tol=1e-8):
    """mu_i = 0 para restricciones inactivas; para las activas, minimos cuadrados
    de  grad f + Jg_A^T mu_A = 0."""
    act = np.abs(g(x)) < tol
    mu = np.zeros(len(act))
    if act.any():
        mu[act] = np.linalg.lstsq(Jg(x)[act].T, -grad(x), rcond=None)[0]
    return mu


if __name__ == "__main__":
    # ---------- P1: gradientes paralelos, dos puntos KKT (minimo y maximo)
    th = np.linspace(0, 2 * np.pi, 400)
    r = np.sqrt(2)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    xs = np.linspace(-2.2, 2.2, 200)
    X, Y = np.meshgrid(xs, xs)
    ax.contour(X, Y, X + Y, levels=np.arange(-4, 4.1, 0.5), cmap="coolwarm", linewidths=0.8)
    ax.plot(r * np.cos(th), r * np.sin(th), "k", lw=2, label=r"$h(x)=0$")
    for p, name, col in ((np.array([-1., -1.]), "mínimo", "tab:blue"), (np.array([1., 1.]), "máximo", "tab:red")):
        ax.plot(*p, "o", color=col, ms=9)
        ax.annotate("", xy=p + 0.7 * np.array([1, 1]) / np.sqrt(2), xytext=p,
                    arrowprops=dict(arrowstyle="->", color="green", lw=2))
        ax.annotate("", xy=p + 0.45 * p / np.linalg.norm(p), xytext=p,
                    arrowprops=dict(arrowstyle="->", color="purple", lw=2))
    ax.text(-1.85, -0.55, r"$\nabla f$", color="green", fontsize=13)
    ax.text(-1.85, -1.75, r"$\nabla h$", color="purple", fontsize=13)
    ax.text(-1.55, -1.25, "mínimo\n$\\lambda=+1/2$", color="tab:blue", fontsize=10, ha="right")
    ax.text(1.1, 0.45, "máximo\n$\\lambda=-1/2$", color="tab:red", fontsize=10)
    ax.set_aspect("equal"); ax.set_xlim(-2.3, 2.3); ax.set_ylim(-2.3, 2.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(alpha=.3)
    ax.set_title("P1: en los puntos KKT, $\\nabla f \\parallel \\nabla h$")
    fig.tight_layout(); fig.savefig(figpath("fig10_kkt_igualdad.png"), dpi=150); plt.close(fig)

    # ---------- P2: region factible, curvas de nivel, cono de gradientes
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs = np.linspace(-2.6, 3.0, 300); ys = np.linspace(-0.8, 4.6, 300)
    X, Y = np.meshgrid(xs, ys)
    F = (X - 2)**2 + (Y - 1)**2
    ax.contour(X, Y, F, levels=[0.25, 1, 2.25, 4, 6.25, 9], cmap="viridis", linewidths=0.8)
    feas = (X**2 - Y <= 0) & (X + Y - 2 <= 0)
    ax.contourf(X, Y, feas.astype(float), levels=[0.5, 1.5], colors=["#cde9c9"], alpha=0.8)
    xp = np.linspace(-1.9, 1.9, 200)
    ax.plot(xp, xp**2, "k", lw=1.8, label=r"$g_1=x_1^2-x_2=0$")
    ax.plot(xs, 2 - xs, "k--", lw=1.8, label=r"$g_2=x_1+x_2-2=0$")
    ax.plot(2, 1, "s", color="gray", ms=8, label="mín. sin restricciones")
    ax.plot(1, 1, "*", color="gold", mec="k", ms=18, zorder=6, label=r"$x^\ast=(1,1)$")
    s = 0.5
    ax.annotate("", xy=(1 - s * 2 / 2, 1), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color="tab:green", lw=2.2))
    ax.annotate("", xy=(1 + s * 2 / 2 * 1.0, 1 - s * 0.5), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2.2))
    ax.annotate("", xy=(1 + s * 0.8, 1 + s * 0.8), xytext=(1, 1), arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2.2))
    ax.text(-1.6, 0.35, r"$\nabla f=(-2,0)$", color="tab:green", fontsize=11)
    ax.text(1.55, 0.55, r"$\nabla g_1$", color="tab:purple", fontsize=11)
    ax.text(1.65, 1.55, r"$\nabla g_2$", color="tab:orange", fontsize=11)
    ax.set_xlim(-2.6, 3.0); ax.set_ylim(-0.8, 4.6); ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("P2: región factible (verde) y gradientes en $x^\\ast$")
    fig.tight_layout(); fig.savefig(figpath("fig10_kkt_desigualdad.png"), dpi=150); plt.close(fig)

    # ---------- verificacion de KKT en x* (con scipy como referencia)
    cons = [{"type": "ineq", "fun": lambda x: -g2(x), "jac": lambda x: -Jg2(x)}]
    res = minimize(f2, [0., 0.], jac=grad2, constraints=cons, method="SLSQP", options={"ftol": 1e-12})
    x_s = res.x
    mu_s = multiplicadores_activos(x_s, grad2, g2, Jg2, tol=1e-6)
    print("SLSQP:", x_s, res.fun, "mu =", mu_s)
    print("residuos KKT (est, fac, dual, comp):", residuos_kkt(x_s, mu_s, grad2, g2, Jg2))

    # ---------- enumeracion de los 4 conjuntos activos de P2
    print("\nEnumeracion de conjuntos activos (P2):")
    filas = []
    for S in ([], [0], [1], [0, 1]):
        def sistema(z, S=S):
            x = z[:2]; mu = np.zeros(2); mu[S] = z[2:]
            res = list(grad2(x) + Jg2(x).T @ mu)
            res += [g2(x)[i] for i in S]
            return res
        z0 = np.array([1.0, 1.0] + [0.5] * len(S))
        z, info, ier, msg = fsolve(sistema, z0, full_output=True)
        x = z[:2]; mu = np.zeros(2); mu[S] = z[2:]
        ok_p = np.all(g2(x) <= 1e-8); ok_d = np.all(mu >= -1e-8)
        filas.append((S, x, mu, g2(x), ok_p, ok_d))
        print(f"  activas={S}: x={np.round(x,4)} mu={np.round(mu,4)} g={np.round(g2(x),4)} factible={ok_p} mu>=0={ok_d}")
    # el otro punto con ambas activas: (-2,4)
    xo = np.array([-2., 4.]); muo = np.linalg.solve(Jg2(xo).T, -grad2(xo))
    print(f"  ambas activas (otra interseccion) x={xo}: mu={np.round(muo,4)}")

    # ---------- sensibilidad: precio sombra  df*/d(eps) = -mu_2 al relajar g2 <= eps
    print("\nSensibilidad (relajar x1+x2<=2+eps):")
    prev = None
    for eps in (-0.1, -0.01, 0.0, 0.01, 0.1):
        c = [{"type": "ineq", "fun": lambda x, e=eps: -(x[0] + x[1] - 2 - e) , "jac": lambda x: -np.array([1., 1.])},
             {"type": "ineq", "fun": lambda x: -(x[0]**2 - x[1]), "jac": lambda x: -np.array([2 * x[0], -1.])}]
        r_ = minimize(f2, [1., 1.], jac=grad2, constraints=c, method="SLSQP", options={"ftol": 1e-14, "maxiter": 200})
        print(f"  eps={eps:+.2f}: f*={r_.fun:.6f}")
        if eps == 0.0: f0 = r_.fun
    for eps in (0.01, -0.01):
        c = [{"type": "ineq", "fun": lambda x, e=eps: -(x[0] + x[1] - 2 - e), "jac": lambda x: -np.array([1., 1.])},
             {"type": "ineq", "fun": lambda x: -(x[0]**2 - x[1]), "jac": lambda x: -np.array([2 * x[0], -1.])}]
        r_ = minimize(f2, [1., 1.], jac=grad2, constraints=c, method="SLSQP", options={"ftol": 1e-14, "maxiter": 200})
        print(f"  pendiente (eps={eps:+}): {(r_.fun - f0) / eps:.4f}   (teoria -mu2 = {-2/3:.4f})")
