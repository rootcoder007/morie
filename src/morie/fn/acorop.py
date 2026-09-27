"""Ant colony optimization for continuous domains, ACO_R (Socha and Dorigo 2008).

Socha, K. and Dorigo, M. (2008). Ant colony optimization for continuous domains. European Journal of Operational Research 185, 1155-1173.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["ant_colony_continuous"]


def ant_colony_continuous(f, bounds, n_ants=None, q=0.2, xi=0.85, n_pop=20, max_iter=200, seed=0):
    r"""The archive holds the k = n_pop best solutions sorted by value. Each ant picks a guide l with probability proportional to w_l, then samples every coordinate from N(s_li, sigma_i^2) with sigma_i = xi times the mean absolute distance from s_l to the other archive members (Socha and Dorigo 2008, eqs. 6-8); the archive keeps the k best of old and new.

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    n_ants : int, optional
        New solutions per iteration (default n_pop).
    q : float
        Locality of the rank weights w_l proportional to exp(-(l - 1)^2 / (2 q^2 k^2)).
    xi : float
        Pheromone evaporation: sigma_i = xi sum_e |s_ei - s_li| / (k - 1).
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
    Socha, K. and Dorigo, M. (2008). Ant colony optimization for continuous domains. European Journal of Operational Research 185, 1155-1173.

    Examples
    --------
    >>> r = ant_colony_continuous(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    m = n if n_ants is None else int(n_ants)
    order = sorted(range(n), key=lambda i: (F[i], i))
    A, AF = [X[i] for i in order], [F[i] for i in order]
    w = [math.exp(-(l_ * l_) / (2 * q * q * n * n)) / (q * n * math.sqrt(2 * math.pi)) for l_ in range(n)]
    wt = 0.0
    for v in w:
        wt += v
    hist = []
    for _ in range(int(max_iter)):
        new, newf = [], []
        for _a in range(m):
            r = rnd.u() * wt
            acc, g = 0.0, n - 1
            for l_ in range(n):
                acc += w[l_]
                if r < acc:
                    g = l_
                    break
            x = []
            for j in range(d):
                s = 0.0
                for e in range(n):
                    s += abs(A[e][j] - A[g][j])
                sig = xi * s / (n - 1)
                x.append(A[g][j] + sig * rnd.n())
            x = clip(x, lo, hi)
            new.append(x)
            newf.append(float(f(x)))
            nfev += 1
        pool, poolf = A + new, AF + newf
        order = sorted(range(len(pool)), key=lambda i: (poolf[i], i))[:n]
        A, AF = [pool[i] for i in order], [poolf[i] for i in order]
        hist.append(AF[0])
    return result("Ant colony optimization (ACO_R)", "ACO_R (Socha and Dorigo 2008)", A[0], AF[0], hist, nfev)


def cheatsheet():
    return "acorop: Ant colony optimization for continuous domains, ACO_R (Socha and Dorigo 2008)"
