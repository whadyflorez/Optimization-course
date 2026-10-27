def backtracking(f, x, fx, g, p, alpha=1.0, c1=1e-4, shrink=0.5):
    """Armijo backtracking: halve alpha until f drops enough."""
    while f(x + alpha * p) > fx + c1 * alpha * (g @ p):
        alpha *= shrink
    return alpha
