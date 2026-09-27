"""Sine cosine algorithm (Mirjalili 2016).

Mirjalili, S. (2016). SCA: a sine cosine algorithm for solving optimization problems. Knowledge-Based Systems 96, 120-133.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["sine_cosine_algorithm"]


def sine_cosine_algorithm(f, bounds, a=2.0, n_pop=20, max_iter=200, seed=0):
    r"""With r1 = a - t a / T, each coordinate moves x + r1 sin(r2) |r3 P - x| when r4 < 1/2 and x + r1 cos(r2) |r3 P - x| otherwise, r2 ~ U(0, 2 pi), r3 ~ U(0, 2), r4 ~ U(0, 1), P the best point so far (Mirjalili 2016, eqs. 3.3-3.4).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    a : float
        r1 falls linearly from a to 0.
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
    Mirjalili, S. (2016). SCA: a sine cosine algorithm for solving optimization problems. Knowledge-Based Systems 96, 120-133.

    Examples
    --------
    >>> r = sine_cosine_algorithm(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    b = argmin(F)
    bx, bf = list(X[b]), F[b]
    hist = []
    T = int(max_iter)
    for t in range(T):
        r1 = a - t * a / T
        for i in range(n):
            x = []
            for j in range(d):
                r2 = 2 * math.pi * rnd.u()
                r3 = 2 * rnd.u()
                r4 = rnd.u()
                tr = math.sin(r2) if r4 < 0.5 else math.cos(r2)
                x.append(X[i][j] + r1 * tr * abs(r3 * bx[j] - X[i][j]))
            X[i] = clip(x, lo, hi)
            F[i] = float(f(X[i]))
            nfev += 1
            if F[i] < bf:
                bx, bf = list(X[i]), F[i]
        hist.append(bf)
    return result("Sine cosine algorithm", "SCA (Mirjalili 2016)", bx, bf, hist, nfev)


def cheatsheet():
    return "scaopt: Sine cosine algorithm (Mirjalili 2016)"
