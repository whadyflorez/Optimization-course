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
