"""Golden-section figures on P1 (pipe insulation)."""
import numpy as np
from style import *
import problems as P, methods as M

f = P.cost_insulation
r = (np.sqrt(5) - 1) / 2
a, b = 0.0, 0.15
rows = []                                    # (a, b, x1, x2, f1, f2) at every iteration
x1, x2 = b - r * (b - a), a + r * (b - a)
f1, f2 = f(np.array([x1])), f(np.array([x2]))
for k in range(8):
    rows.append((a, b, x1, x2, f1, f2))
    if f1 < f2:
        b, x2, f2 = x2, x1, f1; x1 = b - r * (b - a); f1 = f(np.array([x1]))
    else:
        a, x1, f1 = x1, x2, f2; x2 = a + r * (b - a); f2 = f(np.array([x2]))

fig, (ax, bx) = plt.subplots(2, 1, figsize=(7.4, 6.2), height_ratios=[1.0, 1.55], sharex=True)
t = np.linspace(0.002, 0.15, 400)
ax.plot(t * 100, [f(np.array([v])) for v in t], color=BLUE)
ax.set_ylabel("cost [\\$/m/yr]"); ax.set_ylim(20, 120); ax.set_title("Golden-section search on P1: each row is one iteration")
for k, (a_, b_, p1, p2, g1, g2) in enumerate(rows):
    y = -k
    bx.plot([a_ * 100, b_ * 100], [y, y], color=BLUE, lw=5, solid_capstyle="butt", alpha=0.35)
    bx.plot([p1 * 100], [y], "o", color=ORANGE, ms=7)
    bx.plot([p2 * 100], [y], "o", color=AQUA, ms=7)
    bx.text(b_ * 100 + 0.25, y, f"width {100 * (b_ - a_):.2f} cm", va="center", fontsize=8.5, color=GRAY)
    if k < len(rows) - 1:
        ax.axvline(p1 * 100, color=ORANGE, lw=0.0)
bx.set_yticks([-k for k in range(8)]); bx.set_yticklabels([f"k={k}" for k in range(8)])
bx.set_xlabel("insulation thickness $t$ [cm]"); bx.set_xlim(-0.3, 20); bx.grid(axis="y", visible=False)
bx.plot([], [], "o", color=ORANGE, label="$x_1$ (left probe)"); bx.plot([], [], "o", color=AQUA, label="$x_2$ (right probe)")
bx.legend(frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "fig_golden_iter.png")

# convergence: interval width versus number of function evaluations
n = np.arange(2, 25)
fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.semilogy(n, r ** (n - 2), "o-", color=BLUE, ms=4, label="golden section ($0.618$ per evaluation)")
ax.semilogy(n, 0.5 ** ((n - 2) / 2), "s-", color=ORANGE, ms=4, label="halving with 2 new probes per step ($0.707$)")
ax.set_xlabel("function evaluations"); ax.set_ylabel("interval width / initial width")
ax.legend(frameon=False, loc="upper right"); ax.set_title("Reusing one probe per iteration pays off")
save(fig, "fig_golden_conv.png")
