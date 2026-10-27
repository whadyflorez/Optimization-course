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
