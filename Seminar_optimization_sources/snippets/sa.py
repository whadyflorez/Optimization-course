def simulated_annealing(f, x0, T0, alpha, step, n_steps, rng, lo, hi):
    """Metropolis chain whose temperature falls geometrically: T_k = T0 * alpha**k."""
    x, fx = np.array(x0, float), f(x0)
    best_x, best_f = x.copy(), fx
    chain, values, temps = [x.copy()], [fx], [T0]
    T = T0
    for k in range(n_steps):
        x, fx, _ = metropolis_step(f, x, fx, T, step, rng, lo, hi)
        if fx < best_f:
            best_x, best_f = x.copy(), fx      # remember the best
        T *= alpha                         # cooling schedule
        chain.append(x.copy())
        values.append(fx)
        temps.append(T)
    return best_x, best_f, np.array(chain), np.array(values), np.array(temps)
