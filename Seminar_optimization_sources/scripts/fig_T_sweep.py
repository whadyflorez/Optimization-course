"""Sensitivity to the temperature knob: fixed-T Metropolis vs simulated annealing (best-so-far success)."""
import json
import numpy as np
from style import *
import problems as P, methods as M

f2, lo, hi = P.temperature_board, P.LO2, P.HI2
X0, N, STEP, OK, NSEED = np.array([23.0, 8.0]), 2000, 1.0, 26.0, 60
Ts = [0.5, 1, 2, 3, 6, 10, 20, 40]
met = []
for T in Ts:
    r = [M.metropolis(f2, X0, T, STEP, N, np.random.default_rng(s), lo, hi)[1].min() for s in range(NSEED)]
    met.append(float((np.array(r) < OK).mean()))
sa = []
for T0 in Ts:
    r = [M.simulated_annealing(f2, X0, T0, 1 - 6 / N, STEP, N, np.random.default_rng(s), lo, hi)[1] for s in range(NSEED)]
    sa.append(float((np.array(r) < OK).mean()))
json.dump(dict(T=Ts, metropolis=met, sa=sa), open("../tablas/T_sweep.json", "w"))
print(met, sa)
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.semilogx(Ts, 100 * np.array(met), "o-", color=BLUE, label="Metropolis at fixed $T$")
ax.semilogx(Ts, 100 * np.array(sa), "s-", color=ORANGE, label="simulated annealing starting at $T_0=T$")
ax.set_xlabel("temperature $T$ (or $T_0$) [$^\\circ$C]"); ax.set_ylabel("success rate [%]"); ax.set_ylim(-3, 105)
ax.legend(frameon=False, loc="lower center"); ax.set_title(f"How sensitive is the temperature choice? ({NSEED} seeds)")
save(fig, "fig_T_sweep.png")
