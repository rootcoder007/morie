"""Bat algorithm (Yang 2010).

Yang, X.-S. (2010). A new metaheuristic bat-inspired algorithm. In Nature Inspired Cooperative Strategies for Optimization (NICSO 2010), Studies in Computational Intelligence 284, 65-74.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["bat_algorithm"]


def bat_algorithm(f, bounds, fmin=0.0, fmax=2.0, A0=1.0, r0=0.5, alpha=0.9, gamma=0.9, n_pop=20, max_iter=200, seed=0):
    r"""Each bat draws a frequency f_i = fmin + (fmax - fmin) beta, updates v_i <- v_i + (x_i - x*) f_i and x_i <- x_i + v_i; with probability 1 - r_i it instead takes a local walk x* + eps mean(A), eps ~ U(-1, 1) per coordinate. The move is accepted when it does not worsen f_i and a uniform draw is below A_i, after which A_i shrinks and r_i grows (Yang 2010, eqs. 2-6).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    fmin, fmax : float
        Frequency range.
    A0, r0 : float
        Initial loudness and pulse rate.
    alpha, gamma : float
        Loudness decay A <- alpha A and pulse growth r = r0 (1 - exp(-gamma t)).
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
    Yang, X.-S. (2010). A new metaheuristic bat-inspired algorithm. In Nature Inspired Cooperative Strategies for Optimization (NICSO 2010), Studies in Computational Intelligence 284, 65-74.

    Examples
    --------
    >>> r = bat_algorithm(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    V = [[0.0] * d for _ in range(n)]
    Aud, R = [float(A0)] * n, [float(r0)] * n
    b = argmin(F)
    bx, bf = list(X[b]), F[b]
    hist = []
    for t in range(1, int(max_iter) + 1):
        for i in range(n):
            fr = fmin + (fmax - fmin) * rnd.u()
            V[i] = [V[i][j] + (X[i][j] - bx[j]) * fr for j in range(d)]
            x = clip([X[i][j] + V[i][j] for j in range(d)], lo, hi)
            if rnd.u() > R[i]:
                am = 0.0
                for v in Aud:
                    am += v
                am /= n
                x = clip([bx[j] + (2 * rnd.u() - 1) * am for j in range(d)], lo, hi)
            fx = float(f(x))
            nfev += 1
            if fx <= F[i] and rnd.u() < Aud[i]:
                X[i], F[i] = x, fx
                Aud[i] *= alpha
                R[i] = r0 * (1 - math.exp(-gamma * t))
            if fx <= bf:
                bx, bf = list(x), fx
        hist.append(bf)
    return result("Bat algorithm", "Bat algorithm (Yang 2010)", bx, bf, hist, nfev)


def cheatsheet():
    return "batalg: Bat algorithm (Yang 2010)"
