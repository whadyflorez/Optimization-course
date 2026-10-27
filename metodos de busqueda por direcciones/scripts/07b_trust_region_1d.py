"""
Trust region en UNA variable (version didactica del script 07).

En 1D todo se puede dibujar en una sola grafica por iteracion:
  - la funcion real f(x)                       (curva azul)
  - el modelo cuadratico m_k(p) alrededor de x_k (parabola naranja)
  - la region de confianza [x_k - Delta, x_k + Delta]  (franja verde)
  - el paso elegido y el punto nuevo            (estrella roja)

Subproblema en 1D:   min_p  m(p) = f(x_k) + g*p + 0.5*h*p^2    s.a. |p| <= Delta
con g = f'(x_k), h = f''(x_k). La constante f(x_k) no cambia el minimizador
(por eso el codigo solo usa g*p + 0.5*h*p^2 para decidir el paso y para
m(0) - m(p), donde se cancela). Se resuelve a mano:
  - si h > 0 y |p_N| <= Delta, con p_N = -g/h:  p = p_N   (paso de Newton, cabe)
  - si h > 0 y |p_N| >  Delta:                  p = -sign(g)*Delta  (borde)
  - si h <= 0 (curvatura negativa o nula):      p = -sign(g)*Delta  (borde;
    el modelo no tiene minimo: sigue bajando hasta donde llega la region)

Nota: en 1D la direccion de Cauchy (-g) y la de Newton (-g/h) apuntan al
mismo lado, asi que la poligonal dogleg se reduce a un segmento recto:
el paso es simplemente "Newton recortado a la region de confianza".

Actualizacion del radio (igual que en dogleg):
    rho = (f(x) - f(x+p)) / (m(0) - m(p))
    rho < 1/4                    -> Delta = Delta/4
    rho > 3/4 y |p| == Delta     -> Delta = min(2*Delta, Delta_max)
    otro caso                    -> Delta igual
    paso aceptado si rho > 0
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath


def f(x):
    return x**4 - 4 * x**2 + x


def df(x):
    return 4 * x**3 - 8 * x + 1


def d2f(x):
    return 12 * x**2 - 8


def tr_step_1d(g, h, Delta):
    """Resuelve el subproblema de trust region en 1D. Devuelve (p, tipo)."""
    if h > 0:
        pN = -g / h
        if abs(pN) <= Delta:
            return pN, "Newton (cabe en la región)"
        return -np.sign(g) * Delta, "Newton recortado al borde"
    return -np.sign(g) * Delta, "curvatura $\\leq 0$: borde"


def trust_region_1d(x0, Delta0=0.5, Delta_max=3.0, tol=1e-6, max_iter=50):
    x, Delta = float(x0), float(Delta0)
    hist = []
    for k in range(max_iter):
        g, h = df(x), d2f(x)
        if abs(g) < tol:
            break
        p, kind = tr_step_1d(g, h, Delta)
        pred = -(g * p + 0.5 * h * p**2)          # m(0) - m(p)
        actual = f(x) - f(x + p)
        rho = actual / pred if pred > 1e-14 else -1.0
        if rho < 0.25:
            new_Delta, action = 0.25 * Delta, "encoger"
        elif rho > 0.75 and np.isclose(abs(p), Delta):
            new_Delta, action = min(2 * Delta, Delta_max), "crecer"
        else:
            new_Delta, action = Delta, "mantener"
        accepted = rho > 0
        hist.append(dict(k=k, x=x, g=g, h=h, Delta=Delta, p=p, kind=kind,
                         rho=rho, accepted=accepted, action=action,
                         x_new=x + p if accepted else x,
                         new_Delta=new_Delta))
        if accepted:
            x = x + p
        Delta = new_Delta
    return x, hist


def draw_panel(ax, it, xlim=(-2.3, 2.3), ylim=(-6.5, 4)):
    xs = np.linspace(*xlim, 500)
    xk, g, h, D = it["x"], it["g"], it["h"], it["Delta"]
    model = f(xk) + g * (xs - xk) + 0.5 * h * (xs - xk) ** 2
    ax.axvspan(xk - D, xk + D, color="tab:green", alpha=0.18,
               label=r"región de confianza")
    ax.plot(xs, f(xs), color="steelblue", lw=2, label=r"$f(x)$ real")
    ax.plot(xs, model, "--", color="darkorange", lw=2, label=r"modelo $m_k$")
    ax.plot(xk, f(xk), "s", color="black", ms=8, label=r"$x_k$", zorder=5)
    if h > 0:
        xN = xk - g / h
        if xlim[0] < xN < xlim[1]:
            ax.plot(xN, f(xk) + g * (xN - xk) + 0.5 * h * (xN - xk) ** 2, "^",
                    color="purple", ms=9, label="mínimo del modelo (Newton)",
                    zorder=5)
    xp = xk + it["p"]
    ax.plot(xp, f(xp), "*", color="crimson", ms=16, markeredgecolor="black",
            label=r"$x_k+p_k$ (paso)", zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x$")
    verdict = "aceptado" if it["accepted"] else "RECHAZADO"
    ax.set_title(f"iter {it['k']}: $\\Delta$={it['Delta']:.3f}, "
                 f"$\\rho$={it['rho']:.2f} ({verdict}) → {it['action']}",
                 fontsize=10)
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    x0 = 0.4
    x_fin, hist = trust_region_1d(x0, Delta0=0.5)
    print(f"x0={x0}: f''(x0)={d2f(x0):.2f}  (curvatura negativa)")
    print(f"{'k':>2} {'x_k':>8} {'f_k':>9} {'f':>7} {'Delta':>7} {'p':>8} "
          f"{'rho':>7}  paso / radio")
    for it in hist:
        print(f"{it['k']:>2} {it['x']:>8.4f} {f(it['x']):>9.4f} {it['h']:>7.2f} "
              f"{it['Delta']:>7.3f} {it['p']:>8.4f} {it['rho']:>7.3f}  "
              f"{it['kind']} / {it['action']}")
    print(f"\nTrust region 1D (manual): {len(hist)} iteraciones, "
          f"x_final={x_fin:.8f}, f_final={f(x_fin):.6f}, f'={df(x_fin):.1e}")

    res = minimize(lambda z: f(z[0]), [x0], jac=lambda z: np.array([df(z[0])]),
                   hess=lambda z: np.array([[d2f(z[0])]]), method="trust-exact")
    print(f"scipy trust-exact: x*={res.x[0]:.8f}, f*={res.fun:.6f}, iters={res.nit}")

    # ---- Figuras ----
    # Dos figuras de 3 paneles (legibles al proyectar): iteraciones 0-2
    # (curvatura negativa, rechazo, ajuste de radio) y 3-5 (convergencia).
    for name, chunk in (("a", hist[0:3]), ("b", hist[3:6])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
        for ax, it in zip(axes, chunk):
            draw_panel(ax, it)
        axes[0].legend(fontsize=8, loc="upper center")
        axes[0].set_ylabel("valor")
        fig.tight_layout()
        fig.savefig(figpath(f"fig07b_tr1d_iteraciones_{name}.png"), dpi=140)
        plt.close(fig)

    # Radio y ratio a lo largo de las iteraciones
    ks = [it["k"] for it in hist]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].step(ks, [it["Delta"] for it in hist], where="mid", color="teal",
                 marker="o")
    axes[0].set_xlabel("Iteración")
    axes[0].set_ylabel(r"$\Delta_k$")
    axes[0].set_title("Radio de la región de confianza")
    axes[0].grid(alpha=0.3)
    axes[1].plot(ks, [it["rho"] for it in hist], "o-", color="crimson")
    axes[1].axhline(0.75, ls="--", color="gray", label=r"$3/4$: crecer")
    axes[1].axhline(0.25, ls="--", color="gray", label=r"$1/4$: encoger")
    axes[1].set_xlabel("Iteración")
    axes[1].set_ylabel(r"$\rho_k$")
    axes[1].set_title(r"Ratio real/predicho $\rho_k$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figpath("fig07b_tr1d_radio_rho.png"), dpi=140)
    plt.close(fig)
