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
