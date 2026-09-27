"""Cuckoo search via Levy flights (Yang and Deb 2009).

Yang, X.-S. and Deb, S. (2009). Cuckoo search via Levy flights. World Congress on Nature and Biologically Inspired Computing (NaBIC 2009), 210-214.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["cuckoo_search"]


def cuckoo_search(f, bounds, pa=0.25, step=0.01, n_pop=20, max_iter=200, seed=0):
    r"""Each nest proposes x_i + step L (x_i - x*) with a Mantegna Levy vector L (beta = 3/2) and replaces nest i when better; then every coordinate of every nest is, with probability pa, moved by r (x_p1 - x_p2) for two random nests, again kept only when better (Yang and Deb 2009).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    pa : float
        Fraction of nests abandoned per generation.
    step : float
        Levy step scale alpha.
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
    Yang, X.-S. and Deb, S. (2009). Cuckoo search via Levy flights. World Congress on Nature and Biologically Inspired Computing (NaBIC 2009), 210-214.

    Examples
    --------
    >>> r = cuckoo_search(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
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
    for _ in range(int(max_iter)):
        for i in range(n):
            L = levy(rnd, d)
            x = clip([X[i][j] + step * L[j] * (X[i][j] - bx[j]) for j in range(d)], lo, hi)
            fx = float(f(x))
            nfev += 1
            if fx < F[i]:
                X[i], F[i] = x, fx
        P1 = [rnd.idx(n) for _ in range(n)]
        P2 = [rnd.idx(n) for _ in range(n)]
        for i in range(n):
            r = rnd.u()
            x = clip([X[i][j] + (r * (X[P1[i]][j] - X[P2[i]][j]) if rnd.u() < pa else 0.0) for j in range(d)], lo, hi)
            fx = float(f(x))
            nfev += 1
            if fx < F[i]:
                X[i], F[i] = x, fx
        b = argmin(F)
        if F[b] < bf:
            bx, bf = list(X[b]), F[b]
        hist.append(bf)
    return result("Cuckoo search", "Cuckoo search via Levy flights (Yang and Deb 2009)", bx, bf, hist, nfev)


def cheatsheet():
    return "cuckoo: Cuckoo search via Levy flights (Yang and Deb 2009)"
