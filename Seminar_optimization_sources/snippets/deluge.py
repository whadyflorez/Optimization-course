def great_deluge(f, x0, level0, target, step, n_steps, rng, lo, hi):
    """Great Deluge: accept any move below a sinking water level."""
    x, fx = np.array(x0, float), f(x0)
    level = level0                   # start high: almost all moves OK
    rain = (level0 - target) / n_steps   # level drops by 'rain' per step
    chain, values, levels = [x.copy()], [fx], [level]
    for k in range(n_steps):
        x_new = x + step * rng.standard_normal(len(x))
        if np.all(x_new >= lo) and np.all(x_new <= hi):
            f_new = f(x_new)
            if f_new <= level:                             # the only acceptance rule
                x, fx = x_new, f_new
        level -= rain
        chain.append(x.copy())
        values.append(fx)
        levels.append(level)
    return x, fx, np.array(chain), np.array(values), np.array(levels)
