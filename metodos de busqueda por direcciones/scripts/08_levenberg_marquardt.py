"""
Levenberg-Marquardt (LM): método especializado para problemas de MÍNIMOS
CUADRADOS NO LINEALES,
    f(x) = 0.5 * ||r(x)||^2 = 0.5 * sum_i r_i(x)^2,   r: R^n -> R^m

Se puede ver como un "puente" entre descenso de gradiente y Newton, e incluso
como caso particular de trust region con una Hessiana aproximada especial:

  - Gradiente:            grad f(x) = J(x)^T r(x)
  - Hessiana APROXIMADA:  H(x) ~ J(x)^T J(x)   (se descarta el término de
    segundas derivadas de r, exacto solo si r es lineal o los residuos son
    peque\u00f1os -> aproximaci\u00f3n de Gauss-Newton)

Paso de Levenberg-Marquardt (regularizaci\u00f3n del sistema de Gauss-Newton):
    (J^T J + lambda*I) p = -J^T r

  - lambda -> 0:  p tiende al paso de GAUSS-NEWTON (casi-Newton, r\u00e1pido
    cerca del m\u00ednimo si los residuos son peque\u00f1os).
  - lambda -> inf: p tiende a  -(1/lambda) J^T r, es decir a un paso de
    DESCENSO DE GRADIENTE muy corto (robusto pero lento).

lambda juega el mismo papel que 1/Delta en trust region: se ajusta en cada
iteraci\u00f3n seg\u00fan qu\u00e9 tan bien predijo el modelo la reducci\u00f3n real de f
(mismo ratio rho_k del script 07): si el paso mejora f, se acepta y se
reduce lambda (confiar m\u00e1s en Gauss-Newton); si empeora, se rechaza y se
aumenta lambda (retroceder hacia gradiente).

Ejemplo: ajuste de un decaimiento exponencial ruidoso
    y(t) = A * exp(-k*t) + b
a datos sint\u00e9ticos, estimando (A, k, b) -> problema de m\u00ednimos cuadrados
no lineal cl\u00e1sico donde LM es el m\u00e9todo est\u00e1ndar (scipy.optimize.curve_fit
usa LM por defecto).
"""
import numpy as np
from scipy.optimize import least_squares, curve_fit
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath
from _plotting import plot_convergence

rng = np.random.default_rng(0)


def model(t, A, k, b):
    return A * np.exp(-k * t) + b


A_true, k_true, b_true = 5.0, 0.7, 1.0
t_data = np.linspace(0, 6, 40)
y_true = model(t_data, A_true, k_true, b_true)
y_data = y_true + rng.normal(scale=0.15, size=t_data.shape)


def residuals(theta):
    A, k, b = theta
    return model(t_data, A, k, b) - y_data


def jacobian(theta):
    A, k, b = theta
    dA = np.exp(-k * t_data)
    dk = -A * t_data * np.exp(-k * t_data)
    db = np.ones_like(t_data)
    return np.column_stack([dA, dk, db])


def levenberg_marquardt(res_fun, jac_fun, theta0, lam0=1e-2, tol=1e-7, max_iter=100):
    theta = np.array(theta0, dtype=float)
    lam = lam0
    traj = [theta.copy()]
    fvals = [0.5 * np.sum(res_fun(theta) ** 2)]
    for k in range(max_iter):
        r = res_fun(theta)
        J = jac_fun(theta)
        g = J.T @ r
        if np.linalg.norm(g) < tol:
            break
        n = len(theta)
        JTJ = J.T @ J
        f_cur = 0.5 * r @ r

        # intentar con lambda actual; si no mejora, aumentar lambda y reintentar
        for _ in range(30):
            A_lm = JTJ + lam * np.eye(n)
            p = np.linalg.solve(A_lm, -g)
            theta_new = theta + p
            r_new = res_fun(theta_new)
            f_new = 0.5 * r_new @ r_new

            pred_reduction = -(g @ p + 0.5 * p @ JTJ @ p)
            actual_reduction = f_cur - f_new
            rho = actual_reduction / pred_reduction if pred_reduction > 1e-14 else -1.0

            if rho > 0:
                theta = theta_new
                lam = max(lam * 0.5, 1e-12)  # confiar m\u00e1s en Gauss-Newton
                break
            else:
                lam *= 3.0  # retroceder hacia descenso de gradiente
        traj.append(theta.copy())
        fvals.append(0.5 * np.sum(res_fun(theta) ** 2))
    return np.array(traj), fvals, k + 1, lam


if __name__ == "__main__":
    theta0 = [1.0, 0.1, 0.0]  # estimaci\u00f3n inicial deliberadamente mala
    traj, fvals, n_iter, lam_final = levenberg_marquardt(residuals, jacobian, theta0)
    print(f"LM manual: {n_iter} iteraciones, theta_final={traj[-1]} "
          f"(verdadero: [{A_true}, {k_true}, {b_true}]), "
          f"SSE_final={2*fvals[-1]:.4f}, lambda_final={lam_final:.2e}")

    # scipy: least_squares con method='lm' (Levenberg-Marquardt, MINPACK)
    res = least_squares(residuals, theta0, jac=jacobian, method="lm")
    print(f"scipy least_squares(lm): theta*={res.x}, "
          f"SSE={2*0.5*np.sum(res.fun**2):.4f}, nfev={res.nfev}")

    plot_convergence([fvals], ["Levenberg-Marquardt (manual)"],
                      "Convergencia LM: suma de cuadrados de residuos",
                      figpath("fig08_lm_convergencia.png"))

    # --- Trayectoria en el espacio de par\u00e1metros (A, k), proyectando b en su valor \u00f3ptimo ---
    def f_Ak(Ak):
        A, k = Ak
        b_opt = traj[-1][2]
        r = model(t_data, A, k, b_opt) - y_data
        return 0.5 * np.sum(r**2)

    from _plotting import plot_trajectory
    traj_Ak = traj[:, [0, 1]]
    plot_trajectory(f_Ak, traj_Ak, xlim=(0, 6), ylim=(0, 1.0),
                     title="Levenberg-Marquardt: trayectoria en $(A,k)$\n(ajuste exponencial)",
                     fname=figpath("fig08_lm_trayectoria_Ak.png"),
                     minimum=(A_true, k_true))

    # --- Ajuste final vs datos ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(t_data, y_data, s=18, color="gray", label="Datos (con ruido)")
    t_fine = np.linspace(0, 6, 300)
    ax.plot(t_fine, model(t_fine, *theta0), "--", color="crimson", label="Estimaci\u00f3n inicial")
    ax.plot(t_fine, model(t_fine, *traj[-1]), "-", color="teal", linewidth=2,
            label="Ajuste LM final")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y$")
    ax.set_title("Ajuste no lineal $y=A e^{-kt}+b$ v\u00eda Levenberg-Marquardt")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figpath("fig08_lm_ajuste.png"), dpi=150)
    plt.close(fig)
