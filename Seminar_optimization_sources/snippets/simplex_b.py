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
