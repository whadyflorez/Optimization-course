def dogleg_step(g, B, Delta):
    """Step inside the trust region |p| <= Delta along the 'dogleg' path Cauchy -> Newton."""
    p_newton = -np.linalg.solve(B, g)
    if np.linalg.norm(p_newton) <= Delta:                  # Newton step fits: take it
        return p_newton
    p_cauchy = -(g @ g) / (g @ B @ g) * g                  # minimiser along steepest descent
    if np.linalg.norm(p_cauchy) >= Delta:                  # even that is too long: truncate
        return -Delta * g / np.linalg.norm(g)
    d = p_newton - p_cauchy                                # walk from Cauchy towards Newton ...
    a, b, c = d @ d, 2 * p_cauchy @ d, p_cauchy @ p_cauchy - Delta ** 2
    tau = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)      # ... until we hit |p| = Delta
    return p_cauchy + tau * d
