"""Artificial bee colony (Karaboga 2005; Karaboga and Basturk 2007).

Karaboga, D. and Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459-471.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["artificial_bee_colony"]


def artificial_bee_colony(f, bounds, limit=None, n_pop=20, max_iter=200, seed=0):
    r"""Employed bees perturb v_ij = x_ij + phi (x_ij - x_kj), phi ~ U(-1, 1), one random coordinate j and partner k != i, keeping v when it is no worse; onlookers repeat this for sources chosen with probability fit_i / sum fit, fit = 1 / (1 + f) for f >= 0 and 1 + |f| otherwise; the most-tried source beyond ``limit`` failures is re-drawn uniformly (scout).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    limit : int, optional
        Abandonment limit for a food source (default n_pop * dimension).
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
    Karaboga, D. and Basturk, B. (2007). A powerful and efficient algorithm for numerical function optimization: artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459-471.

    Examples
    --------
    >>> r = artificial_bee_colony(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    lim = n * d if limit is None else int(limit)
    trial = [0] * n

    def fit(v):
        return 1.0 / (1.0 + v) if v >= 0 else 1.0 + abs(v)

    def probe(i):
        nonlocal nfev
        k = rnd.other(n, i)
        j = rnd.idx(d)
        phi = 2 * rnd.u() - 1
        v = list(X[i])
        v[j] = X[i][j] + phi * (X[i][j] - X[k][j])
        v = clip(v, lo, hi)
        fv = float(f(v))
        nfev += 1
        if fv <= F[i]:
            X[i], F[i], trial[i] = v, fv, 0
        else:
            trial[i] += 1

    b = argmin(F)
    bx, bf = list(X[b]), F[b]
    hist = []
    for _ in range(int(max_iter)):
        for i in range(n):
            probe(i)
        fits = [fit(v) for v in F]
        tot = 0.0
        for v in fits:
            tot += v
        t = i = 0
        while t < n:
            if rnd.u() < fits[i] / tot:
                probe(i)
                t += 1
            i = (i + 1) % n
        b = argmin(F)
        if F[b] < bf:
            bx, bf = list(X[b]), F[b]
        s = 0
        for q in range(1, n):
            if trial[q] > trial[s]:
                s = q
        if trial[s] > lim:
            X[s] = [lo[j] + rnd.u() * (hi[j] - lo[j]) for j in range(d)]
            F[s] = float(f(X[s]))
            nfev += 1
            trial[s] = 0
            if F[s] < bf:
                bx, bf = list(X[s]), F[s]
        hist.append(bf)
    return result("Artificial bee colony", "ABC (Karaboga and Basturk 2007)", bx, bf, hist, nfev)


def cheatsheet():
    return "abcopt: Artificial bee colony (Karaboga 2005; Karaboga and Basturk 2007)"
