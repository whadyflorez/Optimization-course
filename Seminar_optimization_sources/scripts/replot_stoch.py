import json, numpy as np
from style import *
import matplotlib.pyplot as plt
res = json.load(open("../tablas/stochastic_stats.json")); sw = json.load(open("../tablas/T_sweep.json"))
names = list(res)[::-1]; y = np.arange(len(names))
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.barh(y + 0.19, [100*res[n][0] for n in names], height=0.36, color=GRAY, alpha=0.75, label="best point ever visited")
ax.barh(y - 0.19, [100*res[n][1] for n in names], height=0.36, color=BLUE, label="final position of the walker")
for i, n in enumerate(names):
    ax.text(100*res[n][0]+1.5, i+0.19, f"{100*res[n][0]:.0f}%", va="center", fontsize=9)
    ax.text(100*res[n][1]+1.5, i-0.19, f"{100*res[n][1]:.0f}%", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 112); ax.grid(axis="y", visible=False)
ax.set_xlabel("runs that reached the cold region, $f<26\\,^\\circ$C  [%]   (60 seeds, 2000 evaluations)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
save(fig, "fig_stochastic_compare.png")
Ts, mt, st = sw["T"], sw["metropolis"], sw["sa"]
fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
for ax, idx, ttl in ((axs[0], 0, "best point ever visited"), (axs[1], 1, "final position of the walker")):
    ax.semilogx(Ts, [100*r[idx] for r in mt], "o-", color=BLUE, label="Metropolis at fixed $T$")
    ax.semilogx(Ts, [100*r[idx] for r in st], "s-", color=ORANGE, label="SA starting at $T_0=T$")
    ax.set_title(ttl); ax.set_xlabel("temperature $T$ (or $T_0$) [$^\\circ$C]"); ax.set_ylim(-3, 105)
axs[0].set_ylabel("runs reaching $f<26\\,^\\circ$C [%]")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="upper center", ncol=2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92)); save(fig, "fig_T_sweep.png")
