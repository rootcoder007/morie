"""Jaya algorithm (Rao 2016).

Rao, R. V. (2016). Jaya: a simple and new optimization algorithm for solving constrained and unconstrained optimization problems. International Journal of Industrial Engineering Computations 7, 19-34.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["jaya_algorithm"]


def jaya_algorithm(f, bounds, n_pop=20, max_iter=200, seed=0):
    r"""Every candidate proposes x' = x + r1 (x_best - |x|) - r2 (x_worst - |x|) with r1, r2 ~ U(0, 1) per coordinate and keeps it when better (Rao 2016, eq. 1).

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
    Rao, R. V. (2016). Jaya: a simple and new optimization algorithm for solving constrained and unconstrained optimization problems. International Journal of Industrial Engineering Computations 7, 19-34.

    Examples
    --------
    >>> r = jaya_algorithm(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    hist = []
    for _ in range(int(max_iter)):
        b = argmin(F)
        w = 0
        for i in range(1, n):
            if F[i] > F[w]:
                w = i
        xb, xw = list(X[b]), list(X[w])
        for i in range(n):
            x = clip(
                [X[i][j] + rnd.u() * (xb[j] - abs(X[i][j])) - rnd.u() * (xw[j] - abs(X[i][j])) for j in range(d)],
                lo,
                hi,
            )
            fx = float(f(x))
            nfev += 1
            if fx < F[i]:
                X[i], F[i] = x, fx
        hist.append(F[argmin(F)])
    b = argmin(F)
    return result("Jaya algorithm", "Jaya (Rao 2016)", X[b], F[b], hist, nfev)


def cheatsheet():
    return "jayaop: Jaya algorithm (Rao 2016)"
