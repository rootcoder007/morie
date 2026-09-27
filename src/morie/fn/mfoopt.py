"""Moth-flame optimization (Mirjalili 2015).

Mirjalili, S. (2015). Moth-flame optimization algorithm: a novel nature-inspired heuristic paradigm. Knowledge-Based Systems 89, 228-249.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["moth_flame_optimizer"]


def moth_flame_optimizer(f, bounds, b=1.0, n_pop=20, max_iter=200, seed=0):
    r"""Flames are the best n solutions found so far, sorted; their number falls as round(n - t (n - 1) / T). Moth i spirals around flame min(i, count - 1): M = D exp(b s) cos(2 pi s) + F with D = |F - M| and s ~ U(r, 1), r falling linearly from -1 to -2 (Mirjalili 2015, eqs. 3.12-3.14).

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
    Mirjalili, S. (2015). Moth-flame optimization algorithm: a novel nature-inspired heuristic paradigm. Knowledge-Based Systems 89, 228-249.

    Examples
    --------
    >>> r = moth_flame_optimizer(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    order = sorted(range(n), key=lambda i: (F[i], i))
    FL, FF = [list(X[i]) for i in order], [F[i] for i in order]
    hist = []
    T = int(max_iter)
    for t in range(1, T + 1):
        nf = round(n - t * (n - 1) / T)
        r = -1 - t / T
        for i in range(n):
            fl = FL[min(i, nf - 1)]
            x = []
            for j in range(d):
                s = (r - 1) * rnd.u() + 1
                D = abs(fl[j] - X[i][j])
                x.append(D * math.exp(b * s) * math.cos(2 * math.pi * s) + fl[j])
            X[i] = clip(x, lo, hi)
            F[i] = float(f(X[i]))
            nfev += 1
        pool, poolf = FL + [list(x) for x in X], FF + list(F)
        order = sorted(range(len(pool)), key=lambda i: (poolf[i], i))[:n]
        FL, FF = [pool[i] for i in order], [poolf[i] for i in order]
        hist.append(FF[0])
    return result("Moth-flame optimization", "MFO (Mirjalili 2015)", FL[0], FF[0], hist, nfev)


def cheatsheet():
    return "mfoopt: Moth-flame optimization (Mirjalili 2015)"
