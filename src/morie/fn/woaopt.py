"""Whale optimization algorithm (Mirjalili and Lewis 2016).

Mirjalili, S. and Lewis, A. (2016). The whale optimization algorithm. Advances in Engineering Software 95, 51-67.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["whale_optimization"]


def whale_optimization(f, bounds, b=1.0, n_pop=20, max_iter=200, seed=0):
    r"""With a falling from 2 to 0, A = 2 a r - a and C = 2 r: when p < 1/2 a whale encircles the best X* (|A| < 1) or a random whale (|A| >= 1) by X_ref - A |C X_ref - X|; otherwise it follows the spiral |X* - X| exp(b l) cos(2 pi l) + X*, l ~ U(-1, 1) (Mirjalili and Lewis 2016, eqs. 2.1-2.8).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    b : float
        Shape of the logarithmic spiral.
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
    Mirjalili, S. and Lewis, A. (2016). The whale optimization algorithm. Advances in Engineering Software 95, 51-67.

    Examples
    --------
    >>> r = whale_optimization(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
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
        a = 2 - 2 * t / T
        for i in range(n):
            A = 2 * a * rnd.u() - a
            C = 2 * rnd.u()
            p = rnd.u()
            ell = 2 * rnd.u() - 1
            if p < 0.5:
                ref = bx if abs(A) < 1 else X[rnd.idx(n)]
                x = [ref[j] - A * abs(C * ref[j] - X[i][j]) for j in range(d)]
            else:
                x = [abs(bx[j] - X[i][j]) * math.exp(b * ell) * math.cos(2 * math.pi * ell) + bx[j] for j in range(d)]
            X[i] = clip(x, lo, hi)
            F[i] = float(f(X[i]))
            nfev += 1
        for i in range(n):
            if F[i] < bf:
                bx, bf = list(X[i]), F[i]
        hist.append(bf)
    return result("Whale optimization algorithm", "WOA (Mirjalili and Lewis 2016)", bx, bf, hist, nfev)


def cheatsheet():
    return "woaopt: Whale optimization algorithm (Mirjalili and Lewis 2016)"
