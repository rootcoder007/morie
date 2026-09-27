"""Harris hawks optimization (Heidari et al. 2019).

Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H. (2019). Harris hawks optimization: algorithm and applications. Future Generation Computer Systems 97, 849-872.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["harris_hawks_optimizer"]


def harris_hawks_optimizer(f, bounds, n_pop=20, max_iter=200, seed=0):
    r"""Escaping energy E = 2 E0 (1 - t / T), E0 ~ U(-1, 1). Exploration (|E| >= 1): perch on a random hawk, X_rand - r1 |X_rand - 2 r2 X|, or relative to the rabbit and the mean, (X* - X_m) - r3 (lo + r4 (hi - lo)). Exploitation by soft or hard besiege, with or without progressive rapid dives built from Levy flights, as in Heidari et al. (2019, eqs. 1-13).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    n_pop : int
        Population size.
    max_iter : int
        Iterations (generations).
    seed : int
        Philox seed; both language arms reproduce the same run.

    Returns
    -------
    RichResult
        Keys: x, fun (= estimate), history (best value after each iteration), n_fev, method.

    References
    ----------
    Heidari, A. A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. and Chen, H. (2019). Harris hawks optimization: algorithm and applications. Future Generation Computer Systems 97, 849-872.

    Examples
    --------
    >>> r = harris_hawks_optimizer(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    bi = argmin(F)
    bx, bf = list(X[bi]), F[bi]
    hist = []
    T = int(max_iter)
    for t in range(T):
        for i in range(n):
            E0 = 2 * rnd.u() - 1
            E = 2 * E0 * (1 - t / T)
            if abs(E) >= 1:
                q = rnd.u()
                if q >= 0.5:
                    k = rnd.idx(n)
                    r1, r2 = rnd.u(), rnd.u()
                    x = [X[k][j] - r1 * abs(X[k][j] - 2 * r2 * X[i][j]) for j in range(d)]
                else:
                    r3, r4 = rnd.u(), rnd.u()
                    xm = []
                    for j in range(d):
                        s = 0.0
                        for h in range(n):
                            s += X[h][j]
                        xm.append(s / n)
                    x = [(bx[j] - xm[j]) - r3 * (lo[j] + r4 * (hi[j] - lo[j])) for j in range(d)]
                X[i] = clip(x, lo, hi)
                F[i] = float(f(X[i]))
                nfev += 1
            else:
                r = rnd.u()
                J = 2 * (1 - rnd.u())
                if r >= 0.5 and abs(E) >= 0.5:
                    x = [(bx[j] - X[i][j]) - E * abs(J * bx[j] - X[i][j]) for j in range(d)]
                    X[i] = clip(x, lo, hi)
                    F[i] = float(f(X[i]))
                    nfev += 1
                elif r >= 0.5:
                    x = [bx[j] - E * abs(bx[j] - X[i][j]) for j in range(d)]
                    X[i] = clip(x, lo, hi)
                    F[i] = float(f(X[i]))
                    nfev += 1
                else:
                    if abs(E) >= 0.5:
                        base = X[i]
                    else:
                        base = []
                        for j in range(d):
                            s = 0.0
                            for h in range(n):
                                s += X[h][j]
                            base.append(s / n)
                    Y = clip([bx[j] - E * abs(J * bx[j] - base[j]) for j in range(d)], lo, hi)
                    fy = float(f(Y))
                    nfev += 1
                    L = levy(rnd, d)
                    S = [rnd.u() for _ in range(d)]
                    Z = clip([Y[j] + S[j] * L[j] for j in range(d)], lo, hi)
                    fz = float(f(Z))
                    nfev += 1
                    if fy < F[i]:
                        X[i], F[i] = Y, fy
                    elif fz < F[i]:
                        X[i], F[i] = Z, fz
            if F[i] < bf:
                bx, bf = list(X[i]), F[i]
        hist.append(bf)
    return result("Harris hawks optimization", "HHO (Heidari et al. 2019)", bx, bf, hist, nfev)


def cheatsheet():
    return "hhoopt: Harris hawks optimization (Heidari et al. 2019)"
