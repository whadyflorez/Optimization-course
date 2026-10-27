def numjac(g, x, h=1e-6):
    """Jacobian of a vector function g(x) (rows = constraints) by central differences."""
    cols = []
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h * (1 + abs(x[i]))
        cols.append((g(x + e) - g(x - e)) / (2 * e[i]))
    return np.array(cols).T
