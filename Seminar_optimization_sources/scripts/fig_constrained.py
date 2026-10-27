"""Figures for cutting plane and penalty / barrier methods on P4 (solar field)."""
import json
import numpy as np
from style import *
import problems as P, methods as M

f, g = P.cost_solar, P.g_solar
c = np.array([5.0, -2.0]); lo, hi = np.array([0.0, 1.0]), np.array([6.0, 9.0])
XS = np.array([2.7521, 5.9366])
x2 = np.linspace(1, 9, 300)
A_req = np.array([g(np.array([0.0, v]))[0] for v in x2])


def base(ax, xlim=(0, 6), ylim=(1, 9)):
    ax.fill_betweenx(x2, A_req, 7, color=BLUE, alpha=0.12); ax.plot(A_req, x2, color=BLUE, lw=2.2)
    ax.plot(*XS, "*", color=ORANGE, ms=15, mec="white", zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xlabel("$x_1 = A/100\\,\\mathrm{m}^2$"); ax.set_ylabel("$x_2 = \\Delta T/10$ K")


# ---------------------------------------------------------------- cutting plane
x, path, (Ac, bc) = M.cutting_plane(c, g, [0.0, 0.0], lo, hi, tol=1e-4)
print("cutting plane LPs:", len(path) - 1, "x =", x)
gx, gy = np.meshgrid(np.linspace(0, 6, 400), np.linspace(1, 9, 400))
fig, axs = plt.subplots(1, 4, figsize=(13.2, 3.9), sharey=True)
for ax, k in zip(axs, (1, 2, 3, 6)):
    base(ax); mask = np.ones_like(gx, bool)
    for j in range(k):
        mask &= (Ac[j, 0] * gx + Ac[j, 1] * gy <= bc[j] + 1e-12)
        xx = np.linspace(0, 6, 2)
        if abs(Ac[j, 1]) > 1e-9:
            ax.plot(xx, (bc[j] - Ac[j, 0] * xx) / Ac[j, 1], color=GRAY, lw=0.9, ls="--")
    ax.contourf(gx, gy, mask.astype(float), levels=[0.5, 1.5], colors=[VIOLET], alpha=0.16)
    ax.plot(path[1:k + 1, 0], path[1:k + 1, 1], "o-", color=INK, ms=4, lw=0.8)
    ax.plot(*path[k], "o", color=RED, ms=9, zorder=7)
    ax.set_title(f"after {k} cut{'s' if k > 1 else ''}:  $g(x_{k})={g(path[k])[0]:.2f}$")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$" if ax is axs[0] else "")
fig.tight_layout(); save(fig, "fig_cutting_plane.png")

viol = np.array([max(g(p)[0], 1e-6) for p in path]); err = np.linalg.norm(path - XS, axis=1)
fig, ax = plt.subplots(figsize=(5.8, 3.9))
ax.semilogy(viol, "o-", color=ORANGE, label="constraint violation $g(x_k)$")
ax.semilogy(err, "s-", color=BLUE, label="distance to optimum")
ax.set_xlabel("LP number $k$"); ax.legend(frameon=False); ax.set_title("Cutting plane: LPs needed on P4")
save(fig, "fig_cp_conv.png")

# ---------------------------------------------------------------- penalty / barrier 1-D illustration
xx = np.linspace(0.3, 2.2, 400)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
ax = axs[0]
for rho, col in zip((1, 4, 16, 64), (AQUA, BLUE, VIOLET, ORANGE)):
    ax.plot(xx, xx + 0.5 * rho * np.maximum(0, 1 - xx) ** 2, color=col, label=f"$\\rho={rho}$")
ax.plot(xx, xx, color=GRAY, lw=1, ls=":"); ax.axvline(1, color=INK, lw=1); ax.fill_betweenx([-1, 3], 0.3, 1, color=GRAY, alpha=0.12)
ax.set_title("Exterior penalty: minimum sits OUTSIDE"); ax.set_xlabel("$x$   (constraint $x\\geq 1$)"); ax.set_ylabel("penalised cost"); ax.legend(frameon=False)
ax = axs[1]
xb = np.linspace(1.0005, 2.2, 400)
for mu, col in zip((1, 0.4, 0.15, 0.05), (AQUA, BLUE, VIOLET, ORANGE)):
    ax.plot(xb, xb - mu * np.log(xb - 1), color=col, label=f"$\\mu={mu}$")
ax.plot(xx, xx, color=GRAY, lw=1, ls=":"); ax.axvline(1, color=INK, lw=1); ax.fill_betweenx([-1, 3], 0.3, 1, color=GRAY, alpha=0.12)
ax.set_xlim(0.3, 2.2); ax.set_ylim(0.2, 3.0); ax.set_title("Log barrier: minimum sits INSIDE"); ax.set_xlabel("$x$   (constraint $x\\geq 1$)"); ax.legend(frameon=False)
fig.tight_layout(); save(fig, "fig_penalty_1d.png")

# ---------------------------------------------------------------- penalty on P4
rhos = [10, 100, 1e3, 1e4, 1e5]; mus = [1.0, 0.3, 0.1, 0.03, 0.01]
xe = M.solve_sequence(f, g, [2.0, 3.0], rhos, "exterior")
xi = M.solve_sequence(f, g, [5.0, 3.0], mus, "interior")
ge = np.array([g(p)[0] for p in xe]); gi = np.array([g(p)[0] for p in xi])
stats = {"rhos": rhos, "mus": mus, "xe": xe.tolist(), "xi": xi.tolist(), "ge": ge.tolist(), "gi": gi.tolist(),
         "mult_ext": (np.array(rhos) * np.maximum(ge, 0)).tolist(), "mult_int": (-np.array(mus) / gi).tolist(),
         "cp_path": path.tolist(), "cp_g": [g(p)[0] for p in path]}
json.dump(stats, open("../tablas/constrained_stats.json", "w"), indent=1)
print("ext", xe.round(4).tolist(), "\nint", xi.round(4).tolist(), "\nmult", stats["mult_ext"], stats["mult_int"])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 4.0))
ax.axhspan(2.2, XS[0], color=GRAY, alpha=0.15); ax.axhspan(XS[0], 3.3, color=BLUE, alpha=0.10)
ax.axhline(XS[0], color=INK, lw=1)
ax.semilogx(rhos, xe[:, 0], "o-", color=RED, label="exterior penalty ($s=\\rho$)")
ax.semilogx(1 / np.array(mus), xi[:, 0], "s-", color=AQUA, label="log barrier ($s=1/\\mu$)")
ax.text(1.2, 2.27, "infeasible: not enough area", color=GRAY); ax.text(1.2, 3.2, "feasible", color=BLUE)
ax.text(1.2, XS[0] - 0.05, "$x_1^*=2.752$", ha="left", va="top")
ax.set_ylim(2.2, 3.3); ax.set_xlabel("penalty strength $s$"); ax.set_ylabel("collector area $x_1=A/100\\,\\mathrm{m}^2$")
ax.legend(frameon=False, loc="upper right"); ax.set_title("Both sequences squeeze onto the boundary")
bx.loglog(rhos, np.linalg.norm(xe - XS, axis=1), "o-", color=RED, label="exterior: error vs $\\rho$")
bx.loglog(1 / np.array(mus), np.linalg.norm(xi - XS, axis=1), "s-", color=AQUA, label="barrier: error vs $1/\\mu$")
bx.set_xlabel("penalty strength $s$"); bx.set_ylabel("$\\|x-x^*\\|$"); bx.legend(frameon=False)
bx.set_title("Error shrinks like $1/s$")
fig.tight_layout(); save(fig, "fig_penalty_P4.png")
