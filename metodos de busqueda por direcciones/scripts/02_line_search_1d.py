"""
Minimización en una dirección (line search).

Dado x_k y una dirección de descenso d_k (grad(f(x_k))^T d_k < 0), se define
la función de mérito de una variable:
    phi(alpha) = f(x_k + alpha*d_k),   alpha >= 0
y se busca (exacta o aproximadamente) el alpha que la minimiza.

Se ilustran dos estrategias:
  1) Line search EXACTO 1D con scipy.optimize.minimize_scalar (bounded/Brent)
     sobre phi(alpha) directamente.
  2) Backtracking con condición de Armijo (suficiente descenso), que es lo que
     realmente se usa en la práctica porque no requiere resolver phi'(alpha)=0.

     Condición de Armijo:  phi(alpha) <= phi(0) + c1*alpha*phi'(0),  c1 in (0,1)

Se grafica phi(alpha) para una dirección fija desde un punto de la cuadrática
mal condicionada, marcando el alpha exacto y los alphas que Armijo acepta/rechaza
en cada paso de backtracking (mostrando el "achicamiento" alpha, alpha*rho, ...).
"""
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _functions import quad_f, quad_grad


def armijo_backtracking(f, grad, x, d, alpha0=1.0, c1=1e-4, rho=0.5, max_backtracks=30):
    """Backtracking simple: reduce alpha por factor rho hasta cumplir Armijo."""
    phi0 = f(x)
    dphi0 = grad(x) @ d
    alpha = alpha0
    history = []
    for _ in range(max_backtracks):
        phi_alpha = f(x + alpha * d)
        accepted = phi_alpha <= phi0 + c1 * alpha * dphi0
        history.append((alpha, phi_alpha, accepted))
        if accepted:
            break
        alpha *= rho
    return alpha, history


if __name__ == "__main__":
    x0 = np.array([4.0, 0.9])
    d = -quad_grad(x0)
    d = d / np.linalg.norm(d)  # dirección de descenso normalizada

    def phi(alpha):
        return quad_f(x0 + alpha * d)

    # --- Line search exacto (Brent 1D) ---
    res = minimize_scalar(phi, bounds=(0, 6), method="bounded")
    alpha_exact = res.x
    print(f"Line search exacto (Brent):  alpha* = {alpha_exact:.5f}, phi(alpha*) = {res.fun:.5f}")

    # --- Backtracking Armijo ---
    alpha_armijo, hist = armijo_backtracking(quad_f, quad_grad, x0, d, alpha0=6.0)
    print(f"Backtracking Armijo:         alpha_aceptado = {alpha_armijo:.5f}")
    for a, val, ok in hist:
        print(f"    alpha={a:.5f}  phi(alpha)={val:.5f}  {'ACEPTADO' if ok else 'rechazado'}")

    # --- Gráfica de phi(alpha) ---
    alphas = np.linspace(0, 6, 400)
    phis = [phi(a) for a in alphas]
    phi0, dphi0 = phi(0), (phi(1e-6) - phi(0)) / 1e-6

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, phis, label=r"$\varphi(\alpha)=f(x_k+\alpha d_k)$", color="steelblue")
    tangent = phi0 + dphi0 * alphas
    c1 = 1e-4
    armijo_line = phi0 + c1 * dphi0 * alphas
    ax.plot(alphas, armijo_line, "--", color="darkorange",
             label=r"L\'imite de Armijo: $\varphi(0)+c_1\alpha\varphi'(0)$")
    for a, val, ok in hist:
        ax.plot(a, val, "o", color="green" if ok else "red", markersize=8, zorder=5)
    ax.plot(alpha_exact, res.fun, "*", color="gold", markeredgecolor="black",
             markersize=16, label=r"M\'inimo exacto de $\varphi$", zorder=6)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\varphi(\alpha)$")
    ax.set_title("Line search 1D: b\u00fasqueda exacta vs. backtracking (Armijo)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figpath("fig02_line_search_phi.png"), dpi=150)
    plt.close(fig)
