"""Figures for Metropolis, simulated annealing and Great Deluge (P1 and P2)."""
import json
import numpy as np
from matplotlib.collections import LineCollection
from style import *
import problems as P, methods as M
from fig_problems import panel_P2

f2, lo, hi = P.temperature_board, P.LO2, P.HI2
X0, N, STEP, GLOBAL_OK = np.array([23.0, 8.0]), 2000, 1.0, 26.0
TOTAL = 4  # cold start


def base_axes(ax):
    panel_P2(ax, mark=False); ax.set_title(""); ax.get_legend() and ax.get_legend().remove()


def path_on(ax, chain, cmap="viridis", lw=1.1):
    pts = chain.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, linewidth=lw, alpha=0.95)
    lc.set_array(np.arange(len(seg))); ax.add_collection(lc)
    ax.plot(*chain[0], "o", color="white", mec=INK, ms=8, zorder=5)
    ax.plot(*chain[-1], "s", color="white", mec=INK, ms=8, zorder=5)


# ---- 1. acceptance probability ------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.8, 3.9))
dE = np.linspace(0, 30, 300)
for T, c in zip((1, 3, 10, 30), (BLUE, AQUA, ORANGE, VIOLET)):
    ax.plot(dE, np.exp(-dE / T), color=c, label=f"$T={T}\\,^\\circ$C")
ax.axvline(10, color=GRAY, lw=0.8, ls=":")
ax.text(10.3, 0.92, "uphill move of +10 $^\\circ$C", color=GRAY, fontsize=9)
ax.set_xlabel("temperature increase of the proposal  $\\Delta E = f(x')-f(x)$  [$^\\circ$C]")
ax.set_ylabel("acceptance probability $e^{-\\Delta E/T}$"); ax.legend(frameon=False)
save(fig, "fig_metropolis_prob.png")

# ---- 2. Metropolis samples the Boltzmann distribution (P1, 1-D) ----------------------
f1 = P.cost_insulation
tt = np.linspace(0.003, 0.15, 600)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=False)
for ax, T in zip(axs, (2.0, 8.0)):
    rng = np.random.default_rng(1)
    chain, _ = M.metropolis(f1, np.array([0.08]), T, 0.012, 150000, rng, np.array([0.003]), np.array([0.15]))
    ax.hist(chain[5000:, 0] * 100, bins=50, density=True, color=BLUE, alpha=0.35, label="Metropolis chain")
    w = np.exp(-np.array([f1(np.array([v])) for v in tt]) / T)
    w = w / np.trapezoid(w, tt * 100)
    ax.plot(tt * 100, w, color=ORANGE, lw=2.2, label="Boltzmann  $\\propto e^{-f/T}$")
    ax.set_title(f"$T={T:g}$ \\$/m/yr"); ax.set_xlabel("insulation thickness $t$ [cm]")
axs[0].set_ylabel("probability density"); axs[0].legend(frameon=False)
fig.tight_layout(); save(fig, "fig_boltzmann_P1.png")

# ---- 3. fixed-temperature chains on P2 ----------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.7))
for ax, T in zip(axs, (0.01, 3.0, 20.0)):
    base_axes(ax)
    chain, vals = M.metropolis(f2, X0, T, STEP, N, np.random.default_rng(3), lo, hi)
    path_on(ax, chain)
    ax.set_title(f"$T={T:g}$:  final $f={vals[-1]:.1f}\\,^\\circ$C,  best seen $={vals.min():.1f}$")
    ax.set_xlabel("$x_1$ [cm]"); ax.set_ylabel("$x_2$ [cm]" if ax is axs[0] else "")
fig.tight_layout(); save(fig, "fig_metropolis_P2.png")

# ---- 4. simulated annealing run ------------------------------------------------------
ALPHA, T0 = 1 - 6 / N, 20.0
sa_run = None
for seed in range(20):
    out = M.simulated_annealing(f2, X0, T0, ALPHA, STEP, N, np.random.default_rng(seed), lo, hi)
    if out[1] < GLOBAL_OK:
        sa_run, sa_seed = out, seed; break
bx_, bf_, chain, vals, temps = sa_run
fig, ax = plt.subplots(figsize=(6.2, 4.2))
base_axes(ax); path_on(ax, chain, lw=1.0)
ax.plot(*bx_, "*", color="white", mec=INK, ms=17, zorder=6)
ax.set_title(f"SA path (seed {sa_seed}): start = circle, end = square, best = star: {bf_:.1f} $^\\circ$C")
save(fig, "fig_sa_path.png")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
k = np.arange(len(temps))
a1.plot(k, temps, color=ORANGE); a1.set_ylabel("temperature $T_k$"); a1.set_yscale("log"); a1.set_ylim(0.01, 30)
a2.plot(k, vals, color=BLUE, lw=0.9, label="current $f(x_k)$")
a2.plot(k, np.minimum.accumulate(vals), color=INK, lw=1.8, label="best so far")
a2.set_xlabel("iteration $k$"); a2.set_ylabel("board temperature [$^\\circ$C]"); a2.legend(frameon=False, ncol=2)
fig.tight_layout(); save(fig, "fig_sa_schedule.png")

# ---- 5. Great Deluge -----------------------------------------------------------------
L0, TGT = 55.0, 26.0
gd_run = None
for seed in range(40):
    out = M.great_deluge(f2, X0, L0, TGT, STEP, N, np.random.default_rng(seed), lo, hi)
    if out[1] < GLOBAL_OK:
        gd_run, gd_seed = out, seed; break
xg, fg, chain, vals, levels = gd_run
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.0), width_ratios=[1.15, 1])
k = np.arange(len(vals))
a1.plot(k, vals, color=BLUE, lw=0.9, label="current $f(x_k)$ (always below the level)")
a1.plot(k, levels, color=ORANGE, lw=2.2, label="water level")
a1.set_xlabel("iteration $k$"); a1.set_ylabel("board temperature [$^\\circ$C]"); a1.legend(frameon=False, loc="upper right")
base_axes(a2); path_on(a2, chain, lw=0.9); a2.set_title(f"path (seed {gd_seed}); final $f={fg:.1f}\\,^\\circ$C")
fig.tight_layout(); save(fig, "fig_deluge.png")

# ---- 6. head-to-head success rates ---------------------------------------------------
NSEED = 100
def rate(fn):
    res = np.array([fn(np.random.default_rng(s)) for s in range(NSEED)])
    return float((res < GLOBAL_OK).mean()), float(np.mean(res))
methods = {
    "greedy (T = 0)": lambda r: M.metropolis(f2, X0, 0.01, STEP, N, r, lo, hi)[1].min(),
    "Metropolis, fixed $T=3$": lambda r: M.metropolis(f2, X0, 3.0, STEP, N, r, lo, hi)[1][-1],
    "Great Deluge": lambda r: M.great_deluge(f2, X0, L0, TGT, STEP, N, r, lo, hi)[1],
    "Simulated annealing": lambda r: M.simulated_annealing(f2, X0, T0, ALPHA, STEP, N, r, lo, hi)[1],
}
stats = {k: rate(v) for k, v in methods.items()}
json.dump(stats, open("../tablas/stochastic_stats.json", "w"), indent=1)
print(stats)
fig, ax = plt.subplots(figsize=(6.4, 3.7))
names = list(stats); vals_ = [100 * stats[n][0] for n in names]
ax.barh(names[::-1], vals_[::-1], color=[BLUE, AQUA, ORANGE, GRAY][::-1], height=0.55)
for i, v in enumerate(vals_[::-1]):
    ax.text(v + 1.5, i, f"{v:.0f}%", va="center")
ax.set_xlim(0, 108); ax.set_xlabel(f"runs that found the global minimum [%]   ({NSEED} seeds, {N} evaluations each)")
ax.grid(axis="y", visible=False); save(fig, "fig_stochastic_compare.png")
