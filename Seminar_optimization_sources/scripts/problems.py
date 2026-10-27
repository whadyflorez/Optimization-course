"""
Thermal test problems for the seminar.  EVERY objective is f(x) with x a NumPy vector.

P1  Pipe insulation thickness            x = [t]             1-D, unimodal
P2  Where to mount a sensitive part      x = [x1, x2] (cm)   2-D, multimodal (4 local minima)
P3  Heat-recovery exchanger cost         x = [a, m]          2-D, smooth, curved valley
P4  Solar collector field (constrained)  x = [A/100, dT/10]  2-D, convex constraint
"""
import numpy as np
from scipy.special import k0

# ----------------------------------------------------------------------------- P1
# Steam pipe: insulation of thickness t [m] around a pipe of outer radius R1.
R1, T_FLUID, T_AMB = 0.05, 180.0, 20.0          # m, C, C
K_INS, H_OUT = 0.045, 10.0                      # W/mK, W/m2K
HOURS, C_ENERGY = 8000.0, 0.04                  # h/yr, $/kWh
C_VOL, CRF = 2500.0, 0.20                       # $/m3 installed, capital recovery factor


def heat_loss_pipe(x):
    """Heat loss per metre of pipe [W/m];  x = [t]."""
    r2 = R1 + x[0]
    R_th = np.log(r2 / R1) / (2 * np.pi * K_INS) + 1.0 / (2 * np.pi * r2 * H_OUT)
    return (T_FLUID - T_AMB) / R_th


def cost_insulation(x):
    """Annual cost per metre of pipe [$/m/yr] = energy lost + annualised insulation."""
    r2 = R1 + x[0]
    energy = heat_loss_pipe(x) / 1000.0 * HOURS * C_ENERGY
    capital = C_VOL * np.pi * (r2 ** 2 - R1 ** 2) * CRF
    return energy + capital


# ----------------------------------------------------------------------------- P2
# Thin board (fin-type 2-D conduction, convection on both faces).  A point source of q watts gives
#   theta(r) = q / (2 pi k t) * K0(m r);   cooled clamps (heat sinks) are sources with q < 0.
K_PLATE, THK, H_BOARD = 20.0, 0.002, 15.0       # W/mK, m, W/m2K
M_FIN = np.sqrt(2 * H_BOARD / (K_PLATE * THK)) / 100.0     # fin parameter [1/cm]
R0 = 2.0                                        # cm, finite size of chips / clamps
T_BASE = 45.0                                   # C, mean board temperature level
SRC = np.array([[5.0, 5.0, 6.0],                # x [cm], y [cm], q [W]   (q>0 chip, q<0 cooled clamp)
                [12.0, 8.0, 8.0],
                [19.0, 5.0, 6.0],
                [19.0, 12.0, 5.0],
                [5.0, 12.0, 4.0],
                [12.0, 3.0, -9.0],
                [8.5, 12.0, -3.5],
                [15.5, 9.0, -4.0],
                [22.0, 8.0, -3.0]])
LO2, HI2 = np.array([0.0, 0.0]), np.array([24.0, 16.0])   # board 24 x 16 cm


def temperature_board(x):
    """Board temperature [C] at position x = [x1, x2] in cm."""
    d = np.sqrt(np.sum((x - SRC[:, :2]) ** 2, axis=1) + R0 ** 2)
    return T_BASE + np.sum(SRC[:, 2] / (2 * np.pi * K_PLATE * THK) * k0(M_FIN * d))


# ----------------------------------------------------------------------------- P3
# Counter-flow heat-recovery exchanger.  Hot exhaust (C_HOT) heats water (flow m).
# x = [a, m]  with a = A / 50 m2 (normalised area) and m = water flow [kg/s].
CP_W, C_HOT, T_HOT, T_COLD = 4180.0, 20000.0, 250.0, 20.0       # J/kgK, W/K, C, C
U0, M0 = 100.0, 1.0                                             # W/m2K at m = 1 kg/s
C_AREA, C_PUMP, C_HEAT = 60.0, 2.0e4, 120.0                     # $/m2/yr, $/yr/(kg/s)^2.8, $/kW/yr


def exchanger_duty(x):
    """Heat recovered [W] with the effectiveness-NTU relation (counter-flow, Cmin = water side)."""
    A, m = 50.0 * x[0], x[1]
    Cc = m * CP_W
    Cr = Cc / C_HOT
    if A <= 0 or m <= 0 or Cr >= 0.999:                 # outside the model's validity range
        return np.nan
    U = U0 * (m / M0) ** 0.8
    ntu = U * A / Cc
    eff = (1 - np.exp(-ntu * (1 - Cr))) / (1 - Cr * np.exp(-ntu * (1 - Cr)))
    return eff * Cc * (T_HOT - T_COLD)


def cost_exchanger(x):
    """Net annual cost [k$/yr] = area + pumping - value of recovered heat."""
    A, m = 50.0 * x[0], x[1]
    Q = exchanger_duty(x)
    if not np.isfinite(Q):
        return np.inf                                   # infeasible design: infinitely bad
    return (C_AREA * A + C_PUMP * m ** 2.8 - C_HEAT * Q / 1000.0) / 1000.0


# ----------------------------------------------------------------------------- P4
G_SUN, ETA0, A1, A2, Q_DEM = 800.0, 0.78, 3.5, 0.015, 100e3


def eta_collector(dT):
    """Collector efficiency vs mean temperature rise over ambient (quadratic test curve)."""
    return ETA0 - A1 * dT / G_SUN - A2 * dT ** 2 / G_SUN


def cost_solar(x):
    """k$/yr = collector area cost - premium for delivering hotter heat.  x = [A/100 m2, dT/10 K]."""
    return 5.0 * x[0] - 2.0 * x[1]


def g_solar(x):
    """Constraint g(x) <= 0 :  area must cover the demand  A >= Q_dem / (G eta(dT)).  (convex in x)"""
    eta = eta_collector(10.0 * x[1])
    if eta < 0.02:                                      # collector gives no useful heat: infeasible
        return np.array([np.inf])
    return np.array([Q_DEM / (G_SUN * eta) / 100.0 - x[0]])
