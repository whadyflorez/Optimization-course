"""Figures for gradient descent, line search, Newton, dogleg and Nelder-Mead on P3."""
import numpy as np
from matplotlib.patches import Circle, Polygon
from style import *
import problems as P, methods as M

f = P.cost_exchanger
XS = np.array([4.40585873, 1.44066355])
LV = np.array([-93.55, -93.0, -91, -88, -80, -65, -45, -20, 10, 50, 120])
LO, HI = np.array([0.15, 0.2]), np.array([9.5, 3.2])
X0 = np.array([0.5, 0.5])


def contour(ax, lo=LO, hi=HI, n=220):
    X, Y, Z = grid2(f, lo, hi, n)
    Z = np.where(np.isfinite(Z), Z, np.nan)
    ax.contourf(X, Y, Z, levels=LV, cmap="Blues_r", extend="max", alpha=0.85)
    ax.contour(X, Y, Z, levels=LV, colors="white", linewidths=0.35)
    ax.plot(*XS, "*", color=ORANGE, ms=14, mec="white", zorder=6)
    ax.set_xlabel("normalised area $a$"); ax.set_ylabel("water flow $m$ [kg/s]"); ax.grid(False)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])


# ---------------------------------------------------------------- gradient descent
runs = {
    "fixed step $\\alpha=0.01$": M.gradient_descent(f, X0, "fixed", 0.01, 500)[1],
    "fixed step $\\alpha=0.02$ (too big)": M.gradient_descent(f, X0, "fixed", 0.02, 500)[1],
    "line search (Armijo)": M.gradient_descent(f, X0, "line_search", n_iter=1000)[1],
}
fig, ax = plt.subplots(figsize=(7.2, 4.6)); contour(ax)
for (name, p), c in zip(runs.items(), (BLUE, RED, ORANGE)):
    ax.plot(p[:, 0], p[:, 1], "-o", color=c, ms=2.2, lw=1.0, label=f"{name}: {len(p)-1} it.")
ax.plot(*X0, "o", color="white", mec=INK, ms=9, zorder=7)
ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9); ax.set_title("Steepest descent on P3: zig-zag in a narrow valley")
save(fig, "fig_gd_paths.png")

# ---------------------------------------------------------------- line search illustration
g0 = M.numgrad(f, X0); p0 = -g0
al = np.linspace(0, 0.07, 400)
phi = np.array([f(X0 + a * p0) for a in al])
fig, ax = plt.subplots(figsize=(5.9, 4.0))
ax.plot(al, phi, color=BLUE, label="$\\phi(\\alpha)=f(x+\\alpha p)$")
c1 = 0.3   # a visible sufficient-decrease slope (the code uses 1e-4; 0.3 makes the wedge visible)
ax.plot(al, f(X0) + c1 * al * (g0 @ p0), color=GRAY, ls="--", label="Armijo line ($c_1$ enlarged for display)")
a_try, alpha = [], 1.0
while True:
    a_try.append(alpha)
    if f(X0 + alpha * p0) <= f(X0) + c1 * alpha * (g0 @ p0): break
    alpha *= 0.5
for a in a_try[:-1]:
    if a <= 0.07: ax.plot(a, f(X0 + a * p0), "x", color=RED, ms=8, mew=2)
ax.plot(a_try[-1], f(X0 + a_try[-1] * p0), "o", color=AQUA, ms=9)
ax.plot([], [], "x", color=RED, label="rejected trial steps"); ax.plot([], [], "o", color=AQUA, label="accepted step")
ax.set_xlabel("step length $\\alpha$ along $p=-\\nabla f$"); ax.set_ylabel("cost [k\\$/yr]"); ax.set_ylim(-100, 60)
ax.legend(frameon=False, fontsize=9, loc="upper center"); ax.set_title("Backtracking: halve $\\alpha$ until the decrease is 'enough'")
save(fig, "fig_linesearch.png")
print("trial alphas", a_try)

# ---------------------------------------------------------------- Newton
paths = {
    "GD + line search": M.gradient_descent(f, X0, "line_search", n_iter=1000)[1],
    "Newton": M.newton(f, X0)[1],
    "Dogleg": M.trust_region_dogleg(f, X0)[1],
}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
def err(p): return np.maximum(np.linalg.norm(p - XS, axis=1), 1e-12)
for (name, p), c in zip(paths.items(), (ORANGE, BLUE, AQUA)):
    ax.semilogy(err(p), "-o", ms=2.5, color=c, label=f"{name} ({len(p)-1} it.)")
ax.semilogy(err(runs["fixed step $\\alpha=0.01$"]), color=GRAY, lw=1.2, label="fixed step 0.01 (500 it.)")
ax.set_xlim(0, 260); ax.set_xlabel("iteration"); ax.set_ylabel("distance to optimum $\\|x_k-x^*\\|$")
ax.legend(frameon=False); ax.set_title("Convergence on P3 (same start $x_0=(0.5, 0.5)$)")
save(fig, "fig_conv_P3.png")

# Newton: 1-D quadratic-model illustration on P1
f1 = P.cost_insulation
t = np.linspace(0.004, 0.14, 300) * 100
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(t, [f1(np.array([v / 100])) for v in t], color=BLUE, lw=2.2, label="$f(t)$")
xk = np.array([0.012])
cols = [ORANGE, AQUA, VIOLET]
for j in range(3):
    g, H = M.numgrad(f1, xk)[0], M.numhess(f1, xk)[0, 0]
    tt = np.linspace(max(0.004, xk[0] - 0.03), xk[0] + 0.03, 100)
    model = f1(xk) + g * (tt - xk[0]) + 0.5 * H * (tt - xk[0]) ** 2
    ax.plot(tt * 100, model, color=cols[j], lw=1.6, ls="--", label=f"quadratic model at $t_{j}$")
    ax.plot(xk[0] * 100, f1(xk), "o", color=cols[j], ms=7)
    xk = xk - g / H
ax.plot(xk[0] * 100, f1(xk), "*", color=INK, ms=13, label=f"$t_3={xk[0]*100:.2f}$ cm")
ax.set_ylim(25, 100); ax.set_xlabel("insulation thickness $t$ [cm]"); ax.set_ylabel("cost [\\$/m/yr]")
ax.legend(frameon=False, fontsize=9); ax.set_title("Newton = jump to the minimum of the local parabola")
save(fig, "fig_newton_1d.png")

# Newton fails where the Hessian is indefinite
x0f = np.array([0.3, 0.4])
Hf = M.numhess(f, x0f); gf = M.numgrad(f, x0f)
print("eig H at", x0f, np.linalg.eigvalsh(Hf))
pN = -np.linalg.solve(Hf, gf)
fig, ax = plt.subplots(figsize=(7.2, 4.4)); contour(ax, np.array([-1.2, 0.0]), np.array([9.5, 3.2]))
ax.axhline(0, color=INK, lw=1); ax.axvline(0, color=INK, lw=1)
ax.annotate("", xy=x0f + pN, xytext=x0f, arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.text(0.45, -0.75, "raw Newton step: lands outside\nthe model's validity range", color=RED, fontsize=9)
ps = M.newton(f, x0f)[1]; pd = M.trust_region_dogleg(f, x0f)[1]
ax.plot(ps[:, 0], ps[:, 1], "-o", color=BLUE, ms=3, label=f"Newton + shift + line search ({len(ps)-1} it.)")
ax.plot(pd[:, 0], pd[:, 1], "-s", color=AQUA, ms=3, label=f"dogleg trust region ({len(pd)-1} it.)")
ax.plot(*x0f, "o", color="white", mec=INK, ms=9, zorder=7)
ax.set_xlim(-1.2, 9.5); ax.set_ylim(-1.0, 3.2); ax.legend(frameon=True, framealpha=0.9, loc="upper right", fontsize=9)
ax.set_title("Start with negative curvature ($\\lambda_{\\min}H<0$): pure Newton breaks")
save(fig, "fig_newton_fail.png")

# ---------------------------------------------------------------- dogleg geometry
xg, Dl = np.array([1.2, 0.75]), 0.9
g, H = M.numgrad(f, xg), M.numhess(f, xg)
B = H + max(0, 1e-6 - np.linalg.eigvalsh(H)[0]) * np.eye(2)
pN = -np.linalg.solve(B, g); pU = -(g @ g) / (g @ B @ g) * g; ps_ = M.dogleg_step(g, B, Dl)
fig, ax = plt.subplots(figsize=(6.6, 4.8))
gx, gy = np.meshgrid(np.linspace(-1.4, 2.6, 200), np.linspace(-1.2, 1.6, 200))
model = g[0] * gx + g[1] * gy + 0.5 * (B[0, 0] * gx ** 2 + 2 * B[0, 1] * gx * gy + B[1, 1] * gy ** 2)
ax.contour(gx, gy, model, levels=np.linspace(model.min(), model.max() * 0.15, 14), colors=BLUE, linewidths=0.8, alpha=0.7)
ax.add_patch(Circle((0, 0), Dl, fill=True, fc=ORANGE, alpha=0.10)); ax.add_patch(Circle((0, 0), Dl, fill=False, ec=ORANGE, lw=2))
ax.plot([0, pU[0], pN[0]], [0, pU[1], pN[1]], "-", color=INK, lw=1.2, ls="--", label="dogleg path")
ax.plot(*pU, "o", color=VIOLET, ms=8); ax.text(pU[0] - 0.12, pU[1] - 0.22, "Cauchy point $p^U$", color=VIOLET, ha="right")
ax.plot(*pN, "s", color=BLUE, ms=8); ax.text(pN[0] - 0.05, pN[1] + 0.12, "Newton point $p^B$", color=BLUE, ha="right")
ax.plot(*ps_, "*", color=RED, ms=15, mec="white"); ax.text(ps_[0] + 0.12, ps_[1] - 0.32, "chosen step $p$", color=RED)
ax.annotate("", xy=-g / np.linalg.norm(g) * 0.55, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GRAY))
ax.text(*(-g / np.linalg.norm(g) * 0.55 + np.array([-0.05, -0.2])), "$-\\nabla f$", color=GRAY)
ax.plot(0, 0, "o", color=INK, ms=6); ax.set_aspect("equal"); ax.grid(False)
ax.set_xlim(-1.4, 2.6); ax.set_ylim(-1.2, 1.6); ax.set_xlabel("step $p_1$"); ax.set_ylabel("step $p_2$")
ax.set_title("Dogleg: model contours (blue), trust region (orange)")
save(fig, "fig_dogleg_geometry.png")
print("dogleg", pU, pN, ps_)

# dogleg run with radii
xr, pr, rad = M.trust_region_dogleg(f, x0f)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 3.9), width_ratios=[2.3, 1])
contour(a1, np.array([-0.5, 0.0]), np.array([5.2, 2.2]))
a1.set_xlim(-0.5, 5.2); a1.set_ylim(0.0, 2.2)
for k in range(min(len(pr) - 1, 9)):
    a1.add_patch(Circle(pr[k], rad[k], fill=False, ec=ORANGE, lw=0.9, alpha=0.8))
a1.plot(pr[:, 0], pr[:, 1], "-o", color=INK, ms=3.2, lw=1.2); a1.plot(*x0f, "o", color="white", mec=INK, ms=9, zorder=7)
a1.set_title("Dogleg iterations and trust-region radii"); a1.set_aspect("equal")
a2.step(np.arange(len(rad)), rad, where="post", color=ORANGE); a2.set_xlabel("iteration"); a2.set_ylabel("trust radius $\\Delta_k$")
a2.set_title("Radius shrinks when the model is poor,\ngrows when it is good")
fig.tight_layout(); save(fig, "fig_dogleg_run.png")
print("dogleg run its", len(pr) - 1, "radii", rad.round(2))

# ---------------------------------------------------------------- Nelder-Mead
best, fb, hist = M.nelder_mead(f, X0.copy(), 0.5)
fig, ax = plt.subplots(figsize=(7.2, 4.5)); contour(ax, np.array([0.15, 0.2]), np.array([6.0, 2.6]))
ax.set_xlim(0.15, 6.0); ax.set_ylim(0.2, 2.6)
cm = plt.get_cmap("viridis")
for k, S in enumerate(hist[:22]):
    ax.add_patch(Polygon(S, closed=True, fill=False, ec=cm(k / 22), lw=1.0))
ax.set_title(f"Nelder-Mead simplex on P3 ({len(hist)-1} iterations, first 22 drawn)")
save(fig, "fig_nm_run.png")

# generic sketch of the four moves
fig, axs = plt.subplots(1, 4, figsize=(12.4, 3.2))
W, B1, B2 = np.array([0.0, 0.0]), np.array([2.0, 0.2]), np.array([1.0, 1.7])        # worst, middle, best
c = (B1 + B2) / 2
moves = [("reflection", c + (c - W)), ("expansion", c + 2 * (c - W)), ("contraction", c + 0.5 * (W - c)), ("shrink", None)]
for ax, (title, pt) in zip(axs, moves):
    ax.add_patch(Polygon([W, B1, B2], closed=True, fill=True, fc=BLUE, alpha=0.15, ec=BLUE))
    ax.plot(*W, "o", color=RED, ms=8); ax.text(W[0] - 0.1, W[1] - 0.25, "worst", color=RED, ha="right")
    ax.plot(*B2, "o", color=AQUA, ms=8); ax.text(B2[0], B2[1] + 0.12, "best", color=AQUA, ha="center")
    ax.plot(*B1, "o", color=BLUE, ms=8)
    ax.plot(*c, "+", color=INK, ms=9)
    if pt is not None:
        ax.plot(*pt, "*", color=ORANGE, ms=14); ax.annotate("", xy=pt, xytext=W, arrowprops=dict(arrowstyle="->", color=ORANGE))
    else:
        for q in (B1, W):
            nq = B2 + 0.5 * (q - B2); ax.plot(*nq, "*", color=ORANGE, ms=13)
            ax.annotate("", xy=nq, xytext=q, arrowprops=dict(arrowstyle="->", color=ORANGE))
        ax.add_patch(Polygon([B2, (B2 + B1) / 2, (B2 + W) / 2], closed=True, fill=False, ec=ORANGE, ls="--"))
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlim(-1.2, 5.0); ax.set_ylim(-0.9, 3.4); ax.axis("off")
fig.tight_layout(); save(fig, "fig_nm_ops.png")
