"""
Trust region con método DOGLEG.

IDEA GENERAL DE TRUST REGION (a diferencia de line search):
  - Line search: primero elige una DIRECCIÓN (ej. -grad o Newton), luego
    busca cuánto avanzar (alpha) en esa dirección.
  - Trust region: primero fija una REGIÓN de confianza (bola de radio
    Delta_k alrededor de x_k) donde el modelo cuadrático m_k se considera
    confiable, y resuelve (aprox.) el subproblema:

        min_p  m_k(p) = f(x_k) + grad_k^T p + 0.5 p^T H_k p
        s.a.   ||p|| <= Delta_k

    obteniendo simultáneamente dirección Y magnitud del paso.

EL PASO DOGLEG combina dos candidatos:
  - Paso de Cauchy (p_C): el minimizador de m_k a lo largo de la dirección
    de máximo descenso -grad_k, restringido a la bola:
        p_C = -(grad_k^T grad_k)/(grad_k^T H_k grad_k) * grad_k   (si H_k
        es definida positiva en esa dirección; si no, se usa el borde de
        la región). Es barato, siempre disponible, pero conservador.
  - Paso de Newton (p_N = -H_k^{-1} grad_k): el minimizador exacto de m_k
    SIN restricción de radio. Es el paso "ideal" cuando H_k es def. positiva
    y Delta_k es suficientemente grande, pero puede salirse de la región
    de confianza o no ser de descenso si H_k no es def. positiva.

La trayectoria dogleg p(tau), tau in [0,2], es la poligonal:
    p(tau) = tau * p_C,                         0 <= tau <= 1
    p(tau) = p_C + (tau-1)*(p_N - p_C),          1 <= tau <= 2
es decir: primero va del origen a p_C en línea recta, y luego de p_C a p_N.
Se demuestra (Nocedal & Wright, Lema 4.2) que ||p(tau)|| crece
monótonamente con tau, así que hay un único tau* en el segundo tramo donde
la poligonal cruza la frontera ||p||=Delta_k.

REGLA DEL PASO DOGLEG (para un radio Delta_k dado):
  - Si ||p_N|| <= Delta_k: usar el paso de Newton completo, p_k = p_N
    (cabe dentro de la región; es el mejor paso posible).
  - Si ||p_C|| >= Delta_k: usar p_k = Delta_k * p_C/||p_C||
    (ni el paso de Cauchy cabe; truncar a lo largo de -grad_k).
  - Si no (||p_C|| < Delta_k < ||p_N||): el paso dogleg cruza el segundo
    tramo. Se resuelve la ecuación cuadrática en tau (0<=tau<=1 en la
    parametrización del segundo tramo):
        ||p_C + tau*(p_N - p_C)||^2 = Delta_k^2
    y se toma la raíz positiva -> p_k = p_C + tau*(p_N - p_C).

ACTUALIZACIÓN DEL RADIO DE LA REGIÓN Delta_k (a partir del "ratio de
reducción" rho_k, que mide qué tan bien predijo el modelo cuadrático la
reducción REAL de f):

    rho_k = [f(x_k) - f(x_k+p_k)] / [m_k(0) - m_k(p_k)]
            (reducci\u00f3n real)              (reducci\u00f3n predicha por el modelo)

  - rho_k < 1/4  (predicción mala)      -> ENCOGER: Delta_{k+1} = 0.25*Delta_k
  - rho_k > 3/4  Y  ||p_k||==Delta_k    -> CRECER:  Delta_{k+1} = min(2*Delta_k, Delta_max)
                 (el modelo fue muy fiel y el paso usó todo el radio
                  disponible -> vale la pena permitir pasos más grandes)
  - en otro caso                        -> mantener: Delta_{k+1} = Delta_k

  Aceptación del paso: si rho_k > eta (umbral pequeño, ej. eta=0.1... aquí
  se usa el criterio clásico rho_k > 0): x_{k+1} = x_k + p_k; si no, se
  rechaza el paso (x_{k+1}=x_k) y solo se encoge la región para reintentar.
"""
import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import rosen_f, rosen_grad, rosen_hess
from _plotting import plot_trajectory, plot_convergence


def dogleg_step(g, H, Delta):
    """Calcula el paso dogleg p dado gradiente g, Hessiana H y radio Delta."""
    # --- Paso de Cauchy ---
    gBg = g @ H @ g
    if gBg <= 0:
        # curvatura no positiva a lo largo de -g: ir hasta el borde
        pC = -Delta / np.linalg.norm(g) * g
    else:
        alpha_c = (g @ g) / gBg
        pU = -alpha_c * g  # minimizador irrestricto a lo largo de -g (paso "unconstrained Cauchy")
        if np.linalg.norm(pU) >= Delta:
            pC = -Delta / np.linalg.norm(g) * g
        else:
            pC = pU

    # El dogleg esta definido para H definida positiva (Nocedal & Wright, cap. 4).
    # Si H no lo es, p_N no minimiza el modelo (puede ser un punto silla y el
    # modelo SUBIR): se usa solo el paso de Cauchy, que siempre reduce el modelo.
    if np.linalg.eigvalsh(H).min() <= 0:
        return pC, "Cauchy (H no def. positiva)"

    # --- Paso de Newton ---
    try:
        pN = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        pN = pC  # H singular: recurrir al paso de Cauchy

    if np.linalg.norm(pN) <= Delta:
        return pN, "Newton"

    if np.linalg.norm(pC) >= Delta:
        return Delta / np.linalg.norm(pC) * pC, "Cauchy (truncado)"

    # --- Segmento dogleg: resolver ||pC + tau*(pN-pC)||^2 = Delta^2, tau in [0,1] ---
    d = pN - pC
    a = d @ d
    b = 2 * pC @ d
    c = pC @ pC - Delta**2
    tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)  # raíz positiva (a>0)
    p = pC + tau * d
    return p, f"Dogleg (tau={tau:.2f})"


def trust_region_dogleg(f, grad, hess, x0, Delta0=0.5, Delta_max=3.0,
                         tol=1e-10, max_iter=200):
    x = np.array(x0, dtype=float)
    Delta = Delta0
    traj = [x.copy()]
    fvals = [f(x)]
    deltas = [Delta]
    kinds = []
    for k in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        p, kind = dogleg_step(g, H, Delta)
        kinds.append(kind)

        pred_reduction = -(g @ p + 0.5 * p @ H @ p)  # m(0)-m(p)
        actual_reduction = f(x) - f(x + p)
        rho = actual_reduction / pred_reduction if pred_reduction > 1e-14 else -1.0

        if rho > 0:
            x = x + p  # aceptar paso

        if rho < 0.25:
            Delta = 0.25 * Delta
        elif rho > 0.75 and np.isclose(np.linalg.norm(p), Delta, rtol=1e-6):
            Delta = min(2 * Delta, Delta_max)
        # si no, Delta se mantiene igual

        traj.append(x.copy())
        fvals.append(f(x))
        deltas.append(Delta)
    return np.array(traj), fvals, deltas, kinds, k + 1


if __name__ == "__main__":
    x0 = [-1.5, 2.0]
    traj, fvals, deltas, kinds, n_iter = trust_region_dogleg(
        rosen_f, rosen_grad, rosen_hess, x0, Delta0=0.3, Delta_max=3.0,
        tol=1e-5, max_iter=100)
    print(f"Trust region dogleg (manual): {n_iter} iteraciones, "
          f"x_final={traj[-1]}, f_final={fvals[-1]:.3e}")
    print("Tipos de paso usados en las primeras 15 iteraciones:", kinds[:15])
    print(f"Radio Delta: inicial={deltas[0]:.3f}, m\u00edn={min(deltas):.4f}, "
          f"m\u00e1x={max(deltas):.3f}, final={deltas[-1]:.4f}")

    plot_trajectory(rosen_f, traj, xlim=(-2, 2), ylim=(-1, 3),
                     title="Trust region (dogleg)\nRosenbrock",
                     fname=figpath("fig07_dogleg_rosenbrock.png"),
                     levels=np.geomspace(0.5, 2000, 25), minimum=(1, 1))

    # Evolución del radio de la región de confianza
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(deltas, marker=".", color="teal")
    ax.set_xlabel("Iteraci\u00f3n")
    ax.set_ylabel(r"Radio $\Delta_k$")
    ax.set_title("Evoluci\u00f3n del radio de la regi\u00f3n de confianza")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figpath("fig07_dogleg_radio.png"), dpi=150)
    plt.close(fig)

    plot_convergence([fvals], ["Trust region dogleg"],
                      "Convergencia: trust region dogleg", figpath("fig07_dogleg_convergencia.png"))

    # --- Figura esquemática: paso de Cauchy, paso de Newton, trayectoria dogleg ---
    g = rosen_grad(np.array([-1.2, 1.0]))
    H = rosen_hess(np.array([-1.2, 1.0]))
    Delta_demo = 0.3
    pC_full = -(g @ g) / (g @ H @ g) * g if (g @ H @ g) > 0 else -g
    pN_full = np.linalg.solve(H, -g)
    p_demo, kind_demo = dogleg_step(g, H, Delta_demo)

    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(Delta_demo * np.cos(theta), Delta_demo * np.sin(theta), "--",
            color="gray", label=r"Frontera $\|p\|=\Delta_k$")
    ax.plot([0, pC_full[0]], [0, pC_full[1]], "o-", color="orange", label="Direcci\u00f3n a $p_C$ (Cauchy irrestricto)")
    ax.plot([0, pN_full[0]], [0, pN_full[1]], "o-", color="purple", label="Direcci\u00f3n a $p_N$ (Newton)")
    pC_clamped = pC_full if np.linalg.norm(pC_full) <= Delta_demo else Delta_demo / np.linalg.norm(pC_full) * pC_full
    ax.plot([pC_clamped[0], pN_full[0]], [pC_clamped[1], pN_full[1]], ":",
            color="black", linewidth=1, label="Segmento $p_C \\to p_N$")
    ax.plot(*p_demo, "*", color="crimson", markersize=18, markeredgecolor="black",
            label=f"Paso dogleg elegido\n({kind_demo})", zorder=6)
    ax.plot(0, 0, "s", color="black", markersize=8, label="$x_k$ (origen local)")
    ax.set_aspect("equal")
    ax.set_xlabel("$p_1$")
    ax.set_ylabel("$p_2$")
    ax.set_title("Esquema: trayectoria dogleg e intersecci\u00f3n\ncon la frontera de la regi\u00f3n de confianza")
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(figpath("fig07_dogleg_esquema.png"), dpi=150)
    plt.close(fig)

    res = minimize(rosen_f, x0, jac=rosen_grad, hess=rosen_hess, method="dogleg")
    print(f"\nscipy trust-region 'dogleg': x*={res.x}, f*={res.fun:.3e}, iters={res.nit}")
