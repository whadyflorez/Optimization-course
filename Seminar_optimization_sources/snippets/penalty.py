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
