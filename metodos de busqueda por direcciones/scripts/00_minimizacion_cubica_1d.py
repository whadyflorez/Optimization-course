"""
Minimizacion de una funcion de 1 variable mediante un MODELO CUBICO.

Motivacion: en line search necesitamos resolver, muchas veces por iteracion,
    min_{alpha} phi(alpha) = f(x_k + alpha*d_k)
Bisección / búsqueda áurea (Clase 2) solo usan EVALUACIONES de phi, no su
derivada, y convergen linealmente. Si además tenemos phi'(alpha) disponible
(la tenemos siempre: phi'(alpha) = grad f(x_k+alpha*d_k)^T d_k), podemos
construir un modelo mucho más informado: un POLINOMIO CUBICO que interpola
phi Y phi' en dos puntos, y minimizar ese cúbico en forma cerrada.

Dados dos puntos a, b con phi(a), phi'(a), phi(b), phi'(b) conocidos, existe
un único cúbico
    p(alpha) = A*alpha^3 + B*alpha^2 + C*alpha + D
que coincide con phi y phi' en a y en b (4 condiciones, 4 coeficientes).
Si el intervalo [a,b] "encierra" un mínimo (phi'(a)<0<phi'(b), condición de
bracket), ese cúbico tiene un mínimo local dentro de [a,b] con FORMULA
CERRADA (Nocedal & Wright, ec. 3.59):

    d1 = phi'(a) + phi'(b) - 3*(phi(a)-phi(b))/(a-b)
    d2 = signo(b-a) * sqrt(d1^2 - phi'(a)*phi'(b))
    alpha_c = b - (b-a) * (phi'(b) + d2 - d1) / (phi'(b) - phi'(a) + 2*d2)

Algoritmo (interpolación cúbica con bracket, análogo a bisección pero con
convergencia mucho más rápida porque usa curvatura):
  1. Partir de un bracket [a,b] con phi'(a)<0<phi'(b).
  2. Calcular alpha_c con la fórmula cerrada.
  3. Si phi'(alpha_c) tiene el mismo signo que phi'(a): a <- alpha_c.
     Si no: b <- alpha_c.
  4. Repetir hasta |phi'(alpha_c)| < tol.

Esta es exactamente la estrategia que usan internamente los line search
"serios" (condiciones de Wolfe) para refinar alpha una vez que se tiene un
intervalo que encierra un buen punto -- mucho más barata en evaluaciones
que bisección/áurea porque converge superlinealmente.
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath


def phi(alpha):
    return np.exp(alpha) - 3 * alpha


def dphi(alpha):
    return np.exp(alpha) - 3


ALPHA_STAR = np.log(3.0)  # minimizador exacto: phi'(alpha)=0 -> exp(alpha)=3


def cubic_step(a, b, phi, dphi):
    """Un paso de interpolación cúbica: minimizador cerrado del cúbico
    que interpola (phi,phi') en a y en b."""
    fa, fb = phi(a), phi(b)
    ga, gb = dphi(a), dphi(b)
    d1 = ga + gb - 3.0 * (fa - fb) / (a - b)
    s = np.sign(b - a) if b != a else 1.0
    d2 = s * np.sqrt(max(d1**2 - ga * gb, 0.0))
    alpha_c = b - (b - a) * (gb + d2 - d1) / (gb - ga + 2 * d2)
    return alpha_c


def cubic_interpolation_1d(phi, dphi, a, b, tol=1e-10, max_iter=30):
    assert dphi(a) < 0 < dphi(b), "el bracket inicial debe encerrar un minimo"
    traj = [a, b]
    for k in range(max_iter):
        alpha_c = cubic_step(a, b, phi, dphi)
        g_c = dphi(alpha_c)
        traj.append(alpha_c)
        if abs(g_c) < tol:
            break
        if np.sign(g_c) == np.sign(dphi(a)):
            a = alpha_c
        else:
            b = alpha_c
    return alpha_c, traj, k + 1


if __name__ == "__main__":
    a0, b0 = 0.0, 2.0  # bracket inicial: phi'(0)=-2<0, phi'(2)=exp(2)-3>0
    alpha_final, traj, n_iter = cubic_interpolation_1d(phi, dphi, a0, b0, tol=1e-12)
    print(f"Minimizacion 1D por modelo cubico: {n_iter} iteraciones")
    print(f"alpha_final = {alpha_final:.10f}")
    print(f"alpha* (exacto, ln 3) = {ALPHA_STAR:.10f}")
    print(f"error = {abs(alpha_final - ALPHA_STAR):.3e}")

    # Comparacion con Brent (scipy), que no usa derivadas
    res = minimize_scalar(phi, bracket=(a0, (a0 + b0) / 2, b0))
    print(f"\nscipy Brent (sin derivadas): alpha*={res.x:.10f}, "
          f"nit={res.nit if hasattr(res,'nit') else 'NA'}")

    # --- Figura: phi(alpha), el ultimo modelo cubico ajustado, e iterados ---
    alphas = np.linspace(-0.3, 2.2, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(alphas, phi(alphas), color="steelblue", label=r"$\varphi(\alpha)=e^\alpha-3\alpha$")
    ax.axvline(ALPHA_STAR, linestyle="--", color="gray", label=r"$\alpha^\star=\ln 3$")
    for i, al in enumerate(traj):
        ax.plot(al, phi(al), "o", color="crimson", markersize=6, zorder=5)
        ax.annotate(str(i), (al, phi(al)), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varphi(\alpha)$")
    ax.set_title("Interpolaci\u00f3n c\u00fabica: iterados sobre $\\varphi$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    errors = [abs(al - ALPHA_STAR) for al in traj[2:]]
    ax.plot(range(1, len(errors) + 1), errors, "o-", color="darkorange")
    ax.set_yscale("log")
    ax.set_xlabel("Iteraci\u00f3n")
    ax.set_ylabel(r"$|\alpha_k-\alpha^\star|$ (log)")
    ax.set_title("Convergencia (superlineal)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figpath("fig00_cubico_1d.png"), dpi=150)
    plt.close(fig)

    # --- Figura complementaria: el modelo cubico explicito en la 1ra iteracion ---
    def fit_cubic(a, b, phi, dphi):
        """Resuelve el sistema 4x4 para los coeficientes del cubico que
        interpola phi y phi' en a y b (para VISUALIZAR el modelo)."""
        Amat = np.array([
            [a**3, a**2, a, 1],
            [b**3, b**2, b, 1],
            [3*a**2, 2*a, 1, 0],
            [3*b**2, 2*b, 1, 0],
        ])
        rhs = np.array([phi(a), phi(b), dphi(a), dphi(b)])
        return np.linalg.solve(Amat, rhs)  # [A, B, C, D]

    coeffs = fit_cubic(a0, b0, phi, dphi)
    p_model = np.poly1d(coeffs)
    alpha_c0 = cubic_step(a0, b0, phi, dphi)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    alphas2 = np.linspace(-0.3, 2.3, 400)
    ax.plot(alphas2, phi(alphas2), color="steelblue", linewidth=2,
            label=r"$\varphi(\alpha)=e^\alpha-3\alpha$ (función real)")
    ax.plot(alphas2, p_model(alphas2), "--", color="darkorange", linewidth=2,
            label=r"Modelo cúbico $p(\alpha)$ (interpola $\varphi,\varphi'$ en $a,b$)")
    ax.plot(a0, phi(a0), "s", color="black", markersize=8, label=r"$a$ (bracket)")
    ax.plot(b0, phi(b0), "s", color="dimgray", markersize=8, label=r"$b$ (bracket)")
    ax.plot(alpha_c0, p_model(alpha_c0), "*", color="crimson", markersize=18,
            markeredgecolor="black", label=r"$\alpha_c$ = mínimo del modelo cúbico", zorder=6)
    ax.axvline(ALPHA_STAR, linestyle=":", color="gray", label=r"$\alpha^\star$ (real)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("valor")
    ax.set_ylim(-1, 3)
    ax.set_title("El modelo cúbico coincide con $\\varphi$ y $\\varphi'$\nen $a$ y $b$, y su mínimo ya aproxima bien $\\alpha^\\star$")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(figpath("fig00_cubico_modelo.png"), dpi=150)
    plt.close(fig)
