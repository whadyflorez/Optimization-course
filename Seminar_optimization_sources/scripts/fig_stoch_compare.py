"""Head-to-head on P2 (same start, same 2000-evaluation budget) and sensitivity to the temperature knob.
Success = the point reaches the cold region (f < 26 C; global minimum 24.38 C).  Two readings:
'best visited' (the best point ever seen) and 'final' (where the walker stands when the budget ends)."""
import json
import numpy as np
from style import *
import problems as P, methods as M

f2, lo, hi = P.temperature_board, P.LO2, P.HI2
X0, N, STEP, NSEED, OK = np.array([23.0, 8.0]), 2000, 1.0, 60, 26.0
ALPHA, L0, TGT = 1 - 6 / N, 55.0, 26.0


def rates(fn, n=NSEED):
    b = np.array([fn(np.random.default_rng(s)) for s in range(n)])       # rows: (best, final)
    return float((b[:, 0] < OK).mean()), float((b[:, 1] < OK).mean())


def metro(T):
    def fn(r):
        _, v = M.metropolis(f2, X0, T, STEP, N, r, lo, hi); return v.min(), v[-1]
    return fn


def sa(T0):
    def fn(r):
        _, bf, _, v, _ = M.simulated_annealing(f2, X0, T0, ALPHA, STEP, N, r, lo, hi); return bf, v[-1]
    return fn


def gd(r):
    _, _, _, v, _ = M.great_deluge(f2, X0, L0, TGT, STEP, N, r, lo, hi); return v.min(), v[-1]


res = {"greedy ($T\\to0$)": rates(metro(0.01)), "Metropolis, fixed $T=3$": rates(metro(3.0)),
       "Metropolis, fixed $T=10$": rates(metro(10.0)), "Great Deluge": rates(gd),
       "Simulated annealing ($T_0=20$)": rates(sa(20.0))}
json.dump(res, open("../tablas/stochastic_stats.json", "w"), indent=1); print(res)
names = list(res)[::-1]; y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.barh(y + 0.19, [100 * res[n][0] for n in names], height=0.36, color=GRAY, alpha=0.75, label="best point ever visited")
ax.barh(y - 0.19, [100 * res[n][1] for n in names], height=0.36, color=BLUE, label="final position of the walker")
for i, n in enumerate(names):
    ax.text(100 * res[n][0] + 1.5, i + 0.19, f"{100*res[n][0]:.0f}%", va="center", fontsize=9)
    ax.text(100 * res[n][1] + 1.5, i - 0.19, f"{100*res[n][1]:.0f}%", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
ax.set_xlabel(f"runs that reached the cold region, $f<26\\,^\\circ$C  [%]   ({NSEED} seeds, {N} evaluations)")
ax.legend(frameon=False, loc="lower right", fontsize=9); save(fig, "fig_stochastic_compare.png")

Ts = [0.5, 1, 2, 3, 6, 10, 20, 40]
mt = [rates(metro(T)) for T in Ts]; st = [rates(sa(T)) for T in Ts]
json.dump(dict(T=Ts, metropolis=mt, sa=st), open("../tablas/T_sweep.json", "w")); print(mt, st)
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
for ax, idx, ttl in ((axs[0], 0, "best point ever visited"), (axs[1], 1, "final position of the walker")):
    ax.semilogx(Ts, [100 * r[idx] for r in mt], "o-", color=BLUE, label="Metropolis at fixed $T$")
    ax.semilogx(Ts, [100 * r[idx] for r in st], "s-", color=ORANGE, label="SA starting at $T_0=T$")
    ax.set_title(ttl); ax.set_xlabel("temperature $T$ (or $T_0$) [$^\\circ$C]"); ax.set_ylim(-3, 105)
axs[0].set_ylabel("runs reaching $f<26\\,^\\circ$C [%]"); axs[0].legend(frameon=False, loc="lower right"); fig.tight_layout(); save(fig, "fig_T_sweep.png")
