"""
El criterio de Metrópolis y el papel de la temperatura T (visto en
detalle en la sesión de preparación de esta clase):
  P(aceptar un salto que empeora) = exp(-DeltaE / T)
  - T chica  -> casi solo acepta mejoras (greedy)
  - T grande -> acepta casi cualquier cosa (random walk puro)
Se verifica empíricamente (tipo Monte Carlo) que "sortear u~U(0,1) y
aceptar si u<P" produce exactamente una tasa de aceptación = P.
"""
import numpy as np
import matplotlib.pyplot as plt
from _util import savepath

dE = np.linspace(0, 6, 300)
temperaturas = [0.2, 0.5, 1.0, 1.5, 3.0, 8.0]

fig, ax = plt.subplots(figsize=(7, 5))
for T in temperaturas:
    ax.plot(dE, np.exp(-dE / T), lw=2, label=f"T = {T}")
ax.set_xlabel(r"$\Delta E = f_{new} - f_{cur}$")
ax.set_ylabel(r"$P(\mathrm{aceptar}) = e^{-\Delta E/T}$")
ax.set_title("Criterio de Metrópolis: probabilidad de aceptar un salto peor")
ax.axhline(1.0, color="gray", lw=0.7, ls=":")
ax.legend(title="temperatura")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(savepath("fig02_metropolis_curvas_T.png"), dpi=150)
print("Figura:", savepath("fig02_metropolis_curvas_T.png"))

# --- verificacion empirica del sorteo ---
rng = np.random.default_rng(0)
casos = [(1.0, 0.5), (2.5, 1.5), (0.3, 1.0), (4.0, 2.0)]
print(f"\n{'DeltaE':>7} {'T':>5} {'P teorica':>11} {'tasa observada':>16}")
for dE_i, T_i in casos:
    P_teo = np.exp(-dE_i / T_i)
    u = rng.random(100_000)
    print(f"{dE_i:7.2f} {T_i:5.2f} {P_teo:11.4f} {(u < P_teo).mean():16.4f}")
