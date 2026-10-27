def metropolis_step(f, x, fx, T, step, rng, lo, hi):
    """One Metropolis move: propose, then accept with probability min(1, exp(-dE/T))."""
    x_new = x + step * rng.standard_normal(len(x))        # random proposal
    if np.any(x_new < lo) or np.any(x_new > hi):          # outside the board: reject
        return x, fx, False
    f_new = f(x_new)
    dE = f_new - fx
    if dE <= 0 or rng.random() < np.exp(-dE / T):         # downhill always, uphill sometimes
        return x_new, f_new, True
    return x, fx, False


def metropolis(f, x0, T, step, n_steps, rng, lo, hi):
    """Run a Markov chain at FIXED temperature T."""
    x, fx = np.array(x0, float), f(x0)
    chain, values = [x.copy()], [fx]
    for _ in range(n_steps):
        x, fx, _ = metropolis_step(f, x, fx, T, step, rng, lo, hi)
        chain.append(x.copy())
        values.append(fx)
    return np.array(chain), np.array(values)
