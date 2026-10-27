"""
Optimisation methods for the seminar.  Every objective is f(x) with x a NumPy VECTOR.
Blocks between  '# --8<-- name'  and  '# --8<-- end'  are copied verbatim onto the slides.
"""
import numpy as np
from scipy.optimize import linprog


# --8<-- numgrad
def numgrad(f, x, h=1e-6):
    """Central-difference gradient of a scalar function f(x), x a vector."""
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h * (1 + abs(x[i]))
        g[i] = (f(x + e) - f(x - e)) / (2 * e[i])
    return g


def numhess(f, x, h=1e-4):
    """Hessian by differencing the gradient (symmetrised)."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h * (1 + abs(x[i]))
        H[:, i] = (numgrad(f, x + e) - numgrad(f, x - e)) / (2 * e[i])
    return 0.5 * (H + H.T)
# --8<-- end


# --8<-- golden
def golden_section(f, a, b, tol=1e-4):
    """Minimise a unimodal f over [a, b].  f takes a vector x = [t]."""
    r = (np.sqrt(5) - 1) / 2                    # 0.618...  (golden ratio)
    x1, x2 = b - r * (b - a), a + r * (b - a)   # two interior points
    f1, f2 = f(np.array([x1])), f(np.array([x2]))
    history = [(a, b)]
    while b - a > tol:
        if f1 < f2:                             # minimum lies in [a, x2]
            b, x2, f2 = x2, x1, f1              # old x1 becomes the new x2 (reused!)
            x1 = b - r * (b - a)
            f1 = f(np.array([x1]))
        else:                                   # minimum lies in [x1, b]
            a, x1, f1 = x1, x2, f2              # old x2 becomes the new x1 (reused!)
            x2 = a + r * (b - a)
            f2 = f(np.array([x2]))
        history.append((a, b))
    return 0.5 * (a + b), history
# --8<-- end


# --8<-- metropolis
def metropolis_step(f, x, fx, T, step, rng, lo, hi):
    """One Metropolis move: propose, then accept with probability min(1, exp(-dE/T))."""
    x_new = x + step * rng.standard_normal(len(x))        # random proposal
    if np.any(x_new < lo) or np.any(x_new > hi):          # outside the board: reject
        return x, fx, False
    f_new = f(x_new)
    dE = f_new - fx
    if dE <= 0 or rng.random() < np.exp(-dE / T):         # downhill always, uphill sometimes
        return x_new, f_new, True
    return x, fx, False


def metropolis(f, x0, T, step, n_steps, rng, lo, hi):
    """Run a Markov chain at FIXED temperature T."""
    x, fx = np.array(x0, float), f(x0)
    chain, values = [x.copy()], [fx]
    for _ in range(n_steps):
        x, fx, _ = metropolis_step(f, x, fx, T, step, rng, lo, hi)
        chain.append(x.copy())
        values.append(fx)
    return np.array(chain), np.array(values)
# --8<-- end


# --8<-- sa
def simulated_annealing(f, x0, T0, alpha, step, n_steps, rng, lo, hi):
    """Metropolis chain whose temperature falls geometrically: T_k = T0 * alpha**k."""
    x, fx = np.array(x0, float), f(x0)
    best_x, best_f = x.copy(), fx
    chain, values, temps = [x.copy()], [fx], [T0]
    T = T0
    for k in range(n_steps):
        x, fx, _ = metropolis_step(f, x, fx, T, step, rng, lo, hi)
        if fx < best_f:
            best_x, best_f = x.copy(), fx      # remember the best
        T *= alpha                         # cooling schedule
        chain.append(x.copy())
        values.append(fx)
        temps.append(T)
    return best_x, best_f, np.array(chain), np.array(values), np.array(temps)
# --8<-- end


# --8<-- deluge
def great_deluge(f, x0, level0, target, step, n_steps, rng, lo, hi):
    """Great Deluge: accept any move below a sinking water level."""
    x, fx = np.array(x0, float), f(x0)
    level = level0                   # start high: almost all moves OK
    rain = (level0 - target) / n_steps   # level drops by 'rain' per step
    chain, values, levels = [x.copy()], [fx], [level]
    for k in range(n_steps):
        x_new = x + step * rng.standard_normal(len(x))
        if np.all(x_new >= lo) and np.all(x_new <= hi):
            f_new = f(x_new)
            if f_new <= level:                             # the only acceptance rule
                x, fx = x_new, f_new
        level -= rain
        chain.append(x.copy())
        values.append(fx)
        levels.append(level)
    return x, fx, np.array(chain), np.array(values), np.array(levels)
# --8<-- end


# --8<-- linesearch
def backtracking(f, x, fx, g, p, alpha=1.0, c1=1e-4, shrink=0.5):
    """Armijo backtracking: halve alpha until f drops enough."""
    while f(x + alpha * p) > fx + c1 * alpha * (g @ p):
        alpha *= shrink
    return alpha
# --8<-- end


# --8<-- gd
def gradient_descent(f, x0, rule="line_search", alpha0=0.01, n_iter=200, tol=1e-6):
    """Steepest descent with three step-size rules: 'fixed', 'decay', 'line_search'."""
    x = np.array(x0, float)
    path = [x.copy()]
    for k in range(n_iter):
        g = numgrad(f, x)
        if np.linalg.norm(g) < tol:
            break
        p = -g                                             # steepest-descent direction
        if rule == "fixed":
            alpha = alpha0
        elif rule == "decay":
            alpha = alpha0 / (1 + 0.05 * k)                # prescribed, shrinking step
        else:
            alpha = backtracking(f, x, f(x), g, p)         # step chosen by line search
        x = x + alpha * p
        path.append(x.copy())
        if not np.isfinite(f(x)):                          # left the valid region: stop
            break
    return x, np.array(path)
# --8<-- end


# --8<-- newton
def newton(f, x0, n_iter=50, tol=1e-6, safeguard=True):
    """Newton. safeguard=True: shift H if not PD, plus line search."""
    x = np.array(x0, float)
    path = [x.copy()]
    for k in range(n_iter):
        g, H = numgrad(f, x), numhess(f, x)
        if np.linalg.norm(g) < tol:
            break
        if safeguard:
            lam_min = np.linalg.eigvalsh(H)[0]
            if lam_min < 1e-6:                             # not positive definite
                H = H + (1e-6 - lam_min) * np.eye(len(x))  # shift the spectrum
        p = -np.linalg.solve(H, g)                         # solve  H p = -g
        alpha = backtracking(f, x, f(x), g, p) if safeguard else 1.0
        x = x + alpha * p
        path.append(x.copy())
        if not np.isfinite(f(x)):                          # left the valid region: stop
            break
    return x, np.array(path)
# --8<-- end


# --8<-- dogleg
def dogleg_step(g, B, Delta):
    """Step inside the trust region |p| <= Delta along the 'dogleg' path Cauchy -> Newton."""
    p_newton = -np.linalg.solve(B, g)
    if np.linalg.norm(p_newton) <= Delta:                  # Newton step fits: take it
        return p_newton
    p_cauchy = -(g @ g) / (g @ B @ g) * g                  # minimiser along steepest descent
    if np.linalg.norm(p_cauchy) >= Delta:                  # even that is too long: truncate
        return -Delta * g / np.linalg.norm(g)
    d = p_newton - p_cauchy                                # walk from Cauchy towards Newton ...
    a, b, c = d @ d, 2 * p_cauchy @ d, p_cauchy @ p_cauchy - Delta ** 2
    tau = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)      # ... until we hit |p| = Delta
    return p_cauchy + tau * d
# --8<-- end


# --8<-- trust
def trust_region_dogleg(f, x0, Delta0=0.3, Delta_max=3.0, n_iter=60, tol=1e-6):
    x, Delta = np.array(x0, float), Delta0
    path, radii = [x.copy()], [Delta]
    for k in range(n_iter):
        g, H = numgrad(f, x), numhess(f, x)
        if np.linalg.norm(g) < tol:
            break
        lam_min = np.linalg.eigvalsh(H)[0]
        B = H + max(0.0, 1e-6 - lam_min) * np.eye(len(x))  # make the model convex
        p = dogleg_step(g, B, Delta)
        predicted = -(g @ p + 0.5 * p @ B @ p)             # decrease promised by the model
        actual = f(x) - f(x + p)                           # decrease really obtained
        rho = actual / predicted
        if rho < 0.25:
            Delta *= 0.25                                  # model was poor: shrink the region
        elif rho > 0.75 and np.linalg.norm(p) > 0.99 * Delta:
            Delta = min(2 * Delta, Delta_max)              # model was good: expand it
        if rho > 0.1:                                      # accept the step only if it helped
            x = x + p
        path.append(x.copy())
        radii.append(Delta)
    return x, np.array(path), np.array(radii)
# --8<-- end


# --8<-- simplex_a
def nelder_mead(f, x0, size=0.5, n_iter=100, tol=1e-8):
    """Nelder-Mead simplex: n+1 points, no derivatives."""
    n = len(x0)
    S = np.array([x0] + [x0 + size * np.eye(n)[i] for i in range(n)], float)
    F = np.array([f(s) for s in S])
    history = [S.copy()]
    for _ in range(n_iter):
        order = np.argsort(F)
        S, F = S[order], F[order]                          # best first, worst last
        if abs(F[-1] - F[0]) < tol:
            break
        c = S[:-1].mean(axis=0)                            # centroid of all but the worst
        xr = c + (c - S[-1])                               # reflection
        fr = f(xr)
        if fr < F[0]:                                      # very good: try to go further
            xe = c + 2 * (c - S[-1])                       # expansion
            fe = f(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]:                                   # acceptable: keep the reflection
            S[-1], F[-1] = xr, fr
# --8<-- end
# --8<-- simplex_b
        else:                                              # bad: contract towards the centroid
            xc = c + 0.5 * (S[-1] - c)
            fc = f(xc)
            if fc < F[-1]:
                S[-1], F[-1] = xc, fc
            else:                                          # shrink everything towards the best
                S[1:] = S[0] + 0.5 * (S[1:] - S[0])
                F[1:] = [f(s) for s in S[1:]]
        history.append(S.copy())
    best = np.argmin(F)
    return S[best], F[best], history
# --8<-- end


# --8<-- numjac
def numjac(g, x, h=1e-6):
    """Jacobian of a vector function g(x) (rows = constraints) by central differences."""
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h * (1 + abs(x[i]))
        cols.append((g(x + e) - g(x - e)) / (2 * e[i]))
    return np.array(cols).T
# --8<-- end


# --8<-- cutting
def cutting_plane(c, g, x0, lo, hi, tol=1e-4, max_iter=50):
    """Kelley's cutting plane for  min c.x  s.t. g(x) <= 0  (g convex), lo <= x <= hi."""
    A_cuts, b_cuts = [], []
    x = np.array(x0, float)
    path = [x.copy()]
    for k in range(max_iter):
        gx = g(x)
        if np.max(gx) <= tol and k > 0:                    # LP solution is feasible: done
            break
        J = numjac(g, x)
        for i in range(len(gx)):                           # tangent plane (cut) of each g_i
            A_cuts.append(J[i])
            b_cuts.append(J[i] @ x - gx[i])                # g_i(x_k) + J_i (x - x_k) <= 0
        res = linprog(c, A_ub=np.array(A_cuts), b_ub=np.array(b_cuts),
                      bounds=list(zip(lo, hi)), method="highs")
        x = res.x                                          # new LP vertex
        path.append(x.copy())
    return x, np.array(path), (np.array(A_cuts), np.array(b_cuts))
# --8<-- end


# --8<-- penalty
def penalised(f, g, rho=None, mu=None):
    """Return an UNCONSTRAINED function: exterior penalty (rho) or log-barrier (mu)."""
    if rho is not None:
        return lambda x: f(x) + 0.5 * rho * np.sum(np.maximum(0.0, g(x)) ** 2)
    def barrier(x):
        gx = g(x)
        return f(x) - mu * np.sum(np.log(-gx)) if np.all(gx < 0) else np.inf
    return barrier


def solve_sequence(f, g, x0, params, kind="exterior"):
    """Solve a sequence of unconstrained problems; warm-start each from the previous one."""
    x, out = np.array(x0, float), []
    for p in params:
        F = penalised(f, g, rho=p) if kind == "exterior" else penalised(f, g, mu=p)
        x, _ = newton_inner(F, x)
        out.append(x.copy())
    return np.array(out)
# --8<-- end


def newton_inner(F, x0, n_iter=80, tol=1e-9):
    """Damped Newton that tolerates F = +inf outside its domain (used by the barrier)."""
    x = np.array(x0, float)
    path = [x.copy()]
    for _ in range(n_iter):
        g, H = numgrad(F, x), numhess(F, x)
        if np.linalg.norm(g) < tol:
            break
        lam_min = np.linalg.eigvalsh(H)[0]
        if lam_min < 1e-8:
            H = H + (1e-8 - lam_min + 1e-6) * np.eye(len(x))
        p = -np.linalg.solve(H, g)
        alpha, Fx = 1.0, F(x)
        while not np.isfinite(F(x + alpha * p)) or F(x + alpha * p) > Fx + 1e-4 * alpha * (g @ p):
            alpha *= 0.5
            if alpha < 1e-12:
                return x, np.array(path)
        x = x + alpha * p
        path.append(x.copy())
    return x, np.array(path)
