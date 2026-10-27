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
