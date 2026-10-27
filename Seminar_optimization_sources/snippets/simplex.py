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
