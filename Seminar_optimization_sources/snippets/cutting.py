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
