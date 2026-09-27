"""Teaching-learning-based optimization (Rao, Savsani and Vakharia 2011).

Rao, R. V., Savsani, V. J. and Vakharia, D. P. (2011). Teaching-learning-based optimization: a novel method for constrained mechanical design optimization problems. Computer-Aided Design 43, 303-315.
"""

import math  # noqa: F401

from ._metaheur import Rand, argmin, clip, levy, result, setup  # noqa: F401

__all__ = ["teaching_learning_optimizer"]


def teaching_learning_optimizer(f, bounds, n_pop=20, max_iter=200, seed=0):
    r"""Teacher phase: x' = x + r (x_teacher - T_F mean), T_F = 1 or 2 with equal probability; learner phase: with a random partner k, x' = x + r (x - x_k) when f(x) < f(x_k) and x + r (x_k - x) otherwise; each proposal is kept when better (Rao et al. 2011, eqs. 1-4).

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
    Rao, R. V., Savsani, V. J. and Vakharia, D. P. (2011). Teaching-learning-based optimization: a novel method for constrained mechanical design optimization problems. Computer-Aided Design 43, 303-315.

    Examples
    --------
    >>> r = teaching_learning_optimizer(lambda x: sum(v * v for v in x), [(-5, 5)] * 3, max_iter=300)
    >>> r["fun"] < 1e-3
    True
    """
    rnd = Rand(seed)
    lo, hi, X, F = setup(f, bounds, n_pop, rnd)
    n, d = len(X), len(lo)
    nfev = n
    hist = []
    for _ in range(int(max_iter)):
        for i in range(n):
            tch = argmin(F)
            mean = []
            for j in range(d):
                s = 0.0
                for q in range(n):
                    s += X[q][j]
                mean.append(s / n)
            TF = 1 + (1 if rnd.u() >= 0.5 else 0)
            x = clip([X[i][j] + rnd.u() * (X[tch][j] - TF * mean[j]) for j in range(d)], lo, hi)
            fx = float(f(x))
            nfev += 1
            if fx < F[i]:
                X[i], F[i] = x, fx
            k = rnd.other(n, i)
            if F[i] < F[k]:
                x = clip([X[i][j] + rnd.u() * (X[i][j] - X[k][j]) for j in range(d)], lo, hi)
            else:
                x = clip([X[i][j] + rnd.u() * (X[k][j] - X[i][j]) for j in range(d)], lo, hi)
            fx = float(f(x))
            nfev += 1
            if fx < F[i]:
                X[i], F[i] = x, fx
        hist.append(F[argmin(F)])
    b = argmin(F)
    return result(
        "Teaching-learning-based optimization", "TLBO (Rao, Savsani and Vakharia 2011)", X[b], F[b], hist, nfev
    )


def cheatsheet():
    return "tlbopt: Teaching-learning-based optimization (Rao, Savsani and Vakharia 2011)"
