"""Firefly algorithm (Yang 2009).

Yang, X.-S. (2009). Firefly algorithms for multimodal optimization. In Stochastic Algorithms: Foundations and Applications (SAGA 2009), LNCS 5792, 169-178.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["firefly_algorithm"]


def firefly_algorithm(f, bounds, alpha=0.5, beta0=1.0, gamma=1.0, delta=0.97, n_pop=20, max_iter=200, seed=0):
    r"""For every ordered pair with f_j < f_i, firefly i moves x_i <- x_i + beta0 exp(-gamma r_ij^2)(x_j - x_i) + alpha (u - 1/2) (hi - lo), where r_ij is the distance on the unit-scaled box; alpha decays geometrically (Yang 2009, eqs. 1-4).

    Parameters
    ----------
    f : callable
        Objective to minimise, called on a list of floats.
    bounds : sequence of (low, high)
        Box; every candidate is clipped into it.
    alpha : float
        Initial randomisation weight (times the box width), multiplied by delta each iteration.
    beta0, gamma : float
        Attractiveness beta0 exp(-gamma r^2), r measured on the box scaled to [0, 1].
    delta : float
        Decay of alpha.
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
    Yang, X.-S. (2009). Firefly algorithms for multimodal optimization. In Stochastic Algorithms: Foundations and Applications (SAGA 2009), LNCS 5792, 169-178.

    Examples
    --------
    >>> r = firefly_algorithm(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    b = argmin(F)
    bx, bf = list(X[b]), F[b]
    a = float(alpha)
    hist = []
    for _ in range(int(max_iter)):
        for i in range(n):
            for k in range(n):
                if F[k] < F[i]:
                    r2 = 0.0
                    for j in range(d):
                        z = (X[i][j] - X[k][j]) / (hi[j] - lo[j])
                        r2 += z * z
                    beta = beta0 * math.exp(-gamma * r2)
                    x = clip(
                        [
                            X[i][j] + beta * (X[k][j] - X[i][j]) + a * (rnd.u() - 0.5) * (hi[j] - lo[j])
                            for j in range(d)
                        ],
                        lo,
                        hi,
                    )
                    X[i], F[i] = x, float(f(x))
                    nfev += 1
                    if F[i] < bf:
                        bx, bf = list(x), F[i]
        a *= delta
        hist.append(bf)
    return result("Firefly algorithm", "Firefly algorithm (Yang 2009)", bx, bf, hist, nfev)


def cheatsheet():
    return "fflyop: Firefly algorithm (Yang 2009)"
