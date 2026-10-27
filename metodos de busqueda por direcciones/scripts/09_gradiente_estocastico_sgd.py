"""
Descenso de gradiente ESTOCÁSTICO (SGD).

Contexto: cuando f(x) tiene la forma de una SUMA sobre muchos datos,
    f(x) = (1/N) * sum_{i=1}^{N} f_i(x)          (ej.: pérdida de regresión
                                                    sobre N ejemplos)
el gradiente completo (batch) grad f(x) = (1/N) sum_i grad f_i(x) cuesta
O(N) por iteración. Si N es grande (millones de datos), eso es muy caro.

SGD reemplaza el gradiente completo por un ESTIMADOR RUIDOSO pero barato,
calculado sobre un subconjunto ("minibatch") B_k de tamaño b << N elegido
al azar en cada iteración:

    x_{k+1} = x_k - alpha_k * (1/b) * sum_{i in B_k} grad f_i(x_k)

Puntos clave (se ilustran con regresión lineal por mínimos cuadrados,
donde el óptimo se conoce en forma cerrada para poder medir el error):
  1. Con paso CONSTANTE, SGD no converge al óptimo exacto: la trayectoria
     entra en una "nube" de ruido alrededor del mínimo (varianza del
     estimador de gradiente no desaparece).
  2. Con paso DECRECIENTE (alpha_k = alpha0 / (1+k) o alpha0/sqrt(k)), SGD
     sí converge (en valor esperado) porque el ruido se va apagando junto
     con el paso -- a costa de converger más lento.
  3. Comparación con gradiente BATCH (todo el dataset en cada iteración):
     por iteración, batch da un paso mucho más "limpio", pero cada
     iteración es b/N veces más cara. Se compara "iteraciones" y también
     "número de evaluaciones de gradiente individuales" (equivalente a
     tiempo de cómputo real).
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _util import figpath

rng = np.random.default_rng(1)

# --- Datos sintéticos: regresión lineal y_i = w_true^T x_i + ruido ---
N = 500
d = 2
X = rng.normal(size=(N, d))
w_true = np.array([2.0, -3.0])
noise = rng.normal(scale=1.0, size=N)
y = X @ w_true + noise

# Mínimo exacto (mínimos cuadrados, forma cerrada) para medir el error
w_star, *_ = np.linalg.lstsq(X, y, rcond=None)


def f_i_grad(w, i):
    """Gradiente de la pérdida cuadrática de UN dato i: 0.5*(x_i^T w - y_i)^2."""
    xi, yi = X[i], y[i]
    return (xi @ w - yi) * xi


def full_grad(w):
    return (X.T @ (X @ w - y)) / N


def full_loss(w):
    return 0.5 * np.mean((X @ w - y) ** 2)


def sgd(w0, n_epochs, batch_size, alpha_schedule, seed=0):
    """alpha_schedule: función de la época k -> alpha_k."""
    rng_local = np.random.default_rng(seed)
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    losses = [full_loss(w)]
    n_batches = N // batch_size
    for epoch in range(n_epochs):
        idx = rng_local.permutation(N)
        alpha = alpha_schedule(epoch)
        for b in range(n_batches):
            batch = idx[b * batch_size:(b + 1) * batch_size]
            g = np.mean([f_i_grad(w, i) for i in batch], axis=0)
            w = w - alpha * g
        traj.append(w.copy())
        losses.append(full_loss(w))
    return np.array(traj), losses


def batch_gd(w0, n_iter, alpha):
    w = np.array(w0, dtype=float)
    traj = [w.copy()]
    losses = [full_loss(w)]
    for _ in range(n_iter):
        w = w - alpha * full_grad(w)
        traj.append(w.copy())
        losses.append(full_loss(w))
    return np.array(traj), losses


if __name__ == "__main__":
    w0 = [0.0, 0.0]
    n_epochs = 40

    traj_const, loss_const = sgd(w0, n_epochs, batch_size=10,
                                  alpha_schedule=lambda k: 0.05)
    traj_decay, loss_decay = sgd(w0, n_epochs, batch_size=10,
                                  alpha_schedule=lambda k: 0.3 / (1 + k))
    traj_batch, loss_batch = batch_gd(w0, n_epochs, alpha=0.1)

    print(f"w* (\u00f3ptimo exacto, m\u00ednimos cuadrados) = {w_star}")
    print(f"SGD paso constante:   w_final={traj_const[-1]}, "
          f"||w-w*||={np.linalg.norm(traj_const[-1]-w_star):.4f}")
    print(f"SGD paso decreciente: w_final={traj_decay[-1]}, "
          f"||w-w*||={np.linalg.norm(traj_decay[-1]-w_star):.4f}")
    print(f"Gradiente batch:      w_final={traj_batch[-1]}, "
          f"||w-w*||={np.linalg.norm(traj_batch[-1]-w_star):.4f}")

    # --- Trayectorias en el espacio de parámetros (w1, w2) ---
    def f_w(w):
        return full_loss(np.array(w))

    from _plotting import plot_trajectory
    plot_trajectory(f_w, traj_const, xlim=(-1, 3), ylim=(-4, 1),
                     title="SGD, paso constante $\\alpha=0.05$\n(no converge exactamente: 'nube' de ruido)",
                     fname=figpath("fig09_sgd_paso_constante.png"), minimum=tuple(w_star))
    plot_trajectory(f_w, traj_decay, xlim=(-1, 3), ylim=(-4, 1),
                     title="SGD, paso decreciente $\\alpha_k=0.3/(1+k)$\n(converge, m\u00e1s lento al final)",
                     fname=figpath("fig09_sgd_paso_decreciente.png"), minimum=tuple(w_star))
    plot_trajectory(f_w, traj_batch, xlim=(-1, 3), ylim=(-4, 1),
                     title="Gradiente batch (todo el dataset)\ntrayectoria suave",
                     fname=figpath("fig09_batch_gd.png"), minimum=tuple(w_star))

    # --- Comparación de curvas de pérdida ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(loss_const, label="SGD, $\\alpha$ constante", color="crimson")
    ax.plot(loss_decay, label="SGD, $\\alpha_k$ decreciente", color="darkorange")
    ax.plot(loss_batch, label="Gradiente batch", color="steelblue")
    ax.axhline(full_loss(w_star), linestyle="--", color="gray", label="P\u00e9rdida en $w^*$")
    ax.set_yscale("log")
    ax.set_xlabel("\u00c9poca")
    ax.set_ylabel("P\u00e9rdida $f(w)$ (log)")
    ax.set_title("SGD vs. gradiente batch: p\u00e9rdida por \u00e9poca")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figpath("fig09_sgd_vs_batch_perdida.png"), dpi=150)
    plt.close(fig)

    print(f"\nCosto por \u00e9poca: SGD usa {N} evaluaciones de gradiente individual "
          f"(igual que batch), pero actualiza w {N//10} veces por \u00e9poca en vez de 1 "
          "-> por eso baja m\u00e1s r\u00e1pido al inicio en n\u00famero de \u00e9pocas.")
