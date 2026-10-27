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
