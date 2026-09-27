"""Grey wolf optimizer (Mirjalili, Mirjalili and Lewis 2014).

Mirjalili, S., Mirjalili, S. M. and Lewis, A. (2014). Grey wolf optimizer. Advances in Engineering Software 69, 46-61.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["grey_wolf_optimizer"]


def grey_wolf_optimizer(f, bounds, n_pop=20, max_iter=200, seed=0):
    r"""With a falling linearly from 2 to 0, every wolf moves to the mean of X_l - A_l |C_l X_l - X| over the three leaders alpha, beta, delta, with A_l = 2 a r1 - a and C_l = 2 r2 drawn per coordinate (Mirjalili et al. 2014, eqs. 3.1-3.7).

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
    Mirjalili, S., Mirjalili, S. M. and Lewis, A. (2014). Grey wolf optimizer. Advances in Engineering Software 69, 46-61.

    Examples
    --------
    >>> r = grey_wolf_optimizer(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    order = sorted(range(n), key=lambda i: (F[i], i))
    L = [list(X[order[q]]) for q in range(3 if n >= 3 else n)]
    LF = [F[order[q]] for q in range(len(L))]
    while len(L) < 3:
        L.append(list(L[-1]))
        LF.append(LF[-1])
    hist = []
    T = int(max_iter)
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(n):
            x = []
            for j in range(d):
                s = 0.0
                for q in range(3):
                    A = 2 * a * rnd.u() - a
                    C = 2 * rnd.u()
                    s += L[q][j] - A * abs(C * L[q][j] - X[i][j])
                x.append(s / 3)
            X[i] = clip(x, lo, hi)
            F[i] = float(f(X[i]))
            nfev += 1
        for i in range(n):
            if F[i] < LF[0]:
                L = [list(X[i]), L[0], L[1]]
                LF = [F[i], LF[0], LF[1]]
            elif F[i] < LF[1]:
                L = [L[0], list(X[i]), L[1]]
                LF = [LF[0], F[i], LF[1]]
            elif F[i] < LF[2]:
                L[2], LF[2] = list(X[i]), F[i]
        hist.append(LF[0])
    return result("Grey wolf optimizer", "GWO (Mirjalili, Mirjalili and Lewis 2014)", L[0], LF[0], hist, nfev)


def cheatsheet():
    return "gwoopt: Grey wolf optimizer (Mirjalili, Mirjalili and Lewis 2014)"
