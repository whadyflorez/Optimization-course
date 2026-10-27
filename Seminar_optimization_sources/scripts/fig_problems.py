"""Figures for the four thermal test problems (landscape / feasible region)."""
import numpy as np
from style import *
import problems as P


def panel_P1(ax):
    t = np.linspace(0.001, 0.15, 300)
    r2 = P.R1 + t
    energy = np.array([P.heat_loss_pipe(np.array([a])) for a in t]) / 1000 * P.HOURS * P.C_ENERGY
    capital = P.C_VOL * np.pi * (r2 ** 2 - P.R1 ** 2) * P.CRF
    tot = energy + capital
    ax.plot(t * 100, energy, color=ORANGE, label="energy lost")
    ax.plot(t * 100, capital, color=AQUA, label="insulation (annualised)")
    ax.plot(t * 100, tot, color=BLUE, lw=2.6, label="total")
    i = np.argmin(tot)
    ax.plot(t[i] * 100, tot[i], "o", color=INK, ms=7)
    ax.annotate(f"optimum  t = {t[i]*100:.1f} cm\n{tot[i]:.1f} \\$/m/yr", (t[i] * 100, tot[i]),
                xytext=(9.5, 75), arrowprops=dict(arrowstyle="-", color=GRAY))
    ax.set_xlabel("insulation thickness $t$ [cm]"); ax.set_ylabel("annual cost [\\$/m/yr]")
    ax.set_ylim(0, 160); ax.legend(frameon=False, loc="upper right")
    ax.set_title("P1  Pipe insulation (1-D)")


def panel_P2(ax, mark=True):
    X, Y, Z = grid2(P.temperature_board, P.LO2, P.HI2, 160)
    cs = ax.contourf(X, Y, Z, levels=np.arange(20, 72, 4), cmap="YlOrRd", alpha=0.9)
    ax.contour(X, Y, Z, levels=np.arange(20, 72, 4), colors="white", linewidths=0.4)
    chips, clamps = P.SRC[P.SRC[:, 2] > 0], P.SRC[P.SRC[:, 2] < 0]
    ax.plot(chips[:, 0], chips[:, 1], "s", color=INK, ms=8, label="chips (heat sources)")
    ax.plot(clamps[:, 0], clamps[:, 1], "^", color=BLUE, mec="white", ms=9, label="cooled clamps")
    if mark:
        for (a, b) in ((12.0, 2.7), (23.0, 8.1), (15.7, 9.0), (9.0, 12.8)):
            ax.plot(a, b, "o", mfc="none", mec=INK, ms=13, mew=1.6)
    ax.set_xlabel("$x_1$ [cm]"); ax.set_ylabel("$x_2$ [cm]"); ax.set_aspect("equal"); ax.grid(False)
    ax.set_title("P2  Board temperature [$^\\circ$C]")
    return cs


def panel_P3(ax):
    X, Y, Z = grid2(P.cost_exchanger, np.array([0.3, 0.3]), np.array([9.0, 3.0]), 160)
    Z = np.where(np.isfinite(Z), Z, np.nan)
    lv = np.array([-93.6, -93, -91, -88, -80, -65, -45, -20, 10, 50, 120])
    cs = ax.contourf(X, Y, Z, levels=lv, cmap="Blues_r", extend="max")
    ax.contour(X, Y, Z, levels=lv, colors="white", linewidths=0.4)
    ax.plot(4.406, 1.4407, "*", color=ORANGE, ms=15, mec="white")
    ax.set_xlabel("normalised area $a = A/50\\,\\mathrm{m}^2$"); ax.set_ylabel("water flow $m$ [kg/s]")
    ax.set_title("P3  Heat-recovery exchanger cost [k\\$/yr]"); ax.grid(False)
    return cs


def panel_P4(ax):
    x2 = np.linspace(1, 9, 200)
    A_req = np.array([P.g_solar(np.array([0.0, v]))[0] for v in x2])
    ax.fill_betweenx(x2, A_req, 7, color=BLUE, alpha=0.13)
    ax.plot(A_req, x2, color=BLUE, lw=2.4)
    xx = np.array([0, 7])
    for c, col, lw in ((-2.0, GRAY, 0.9), (1.886, ORANGE, 1.4), (5.0, GRAY, 0.9)):
        ax.plot(xx, (5 * xx - c) / 2.0, color=col, lw=lw, ls="--")
    ax.annotate("lower cost  $f=5x_1-2x_2$", xy=(0.55, 6.4), xytext=(0.9, 8.1), color=GRAY,
                arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.plot(2.752, 5.937, "*", color=ORANGE, ms=15, mec="white")
    ax.text(4.3, 2.0, "feasible\n$g(x)\\leq 0$", color=BLUE, ha="center")
    ax.text(0.15, 1.5, "infeasible", color=GRAY)
    ax.set_xlim(0, 7); ax.set_ylim(1, 9)
    ax.set_xlabel("collector area $x_1 = A/100\\,\\mathrm{m}^2$"); ax.set_ylabel("temp. rise $x_2 = \\Delta T/10$ K")
    ax.set_title("P4  Solar field (constrained)")


if __name__ == "__main__":
    for name, fn, size in (("fig_P1_cost.png", panel_P1, (6, 4)), ("fig_P2_landscape.png", panel_P2, (6.4, 4.3)),
                           ("fig_P3_landscape.png", panel_P3, (6, 4.2)), ("fig_P4_feasible.png", panel_P4, (6, 4.2))):
        fig, ax = plt.subplots(figsize=size); fn(ax); save(fig, name)
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
    panel_P1(axs[0, 0]); panel_P2(axs[0, 1], mark=False); panel_P3(axs[1, 0]); panel_P4(axs[1, 1])
    fig.tight_layout(); save(fig, "fig_problems_overview.png")
