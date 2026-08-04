# morie.fn -- function file (rootcoder007/morie)
"""Pareto-smoothed importance weights."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["loo_pareto_smooth"]


def loo_pareto_smooth(log_lik):
    """Replace the largest importance weights by fitted Pareto quantiles.

    Truncating the big weights caps the variance but adds bias; leaving
    them alone keeps it unbiased and useless.  Smoothing does neither:
    the extreme weights are replaced by the order statistics of a
    generalised Pareto fitted to that same tail, which stabilises the
    variance while keeping the shape the data actually showed.

    Formula: with ``M = min(0.2 S, 3 sqrt(S))``, fit a generalised
    Pareto to the ``M`` largest raw weights and substitute
    ``F^-1((z - 0.5) / M)`` for ``z = 1..M``, then cap at
    ``max_w = S^(3/4) mean(w)``.

    Parameters
    ----------
    log_lik : array-like, shape (S, n)
        Pointwise log likelihood.

    Returns
    -------
    RichResult
        ``estimate`` (largest k), ``k``, ``weights`` (normalised, draws
        by observations), ``S``, ``n``.

    References
    ----------
    Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024).
    Pareto smoothed importance sampling.  Journal of Machine Learning
    Research 25:1-58.
    """
    L = C.mat(log_lik)
    Sn = len(L)
    n = len(L[0])
    W = [[0.0] * n for _ in range(Sn)]
    ks = []
    for i in range(n):
        lw, k = S.psis([-L[s][i] for s in range(Sn)])
        ks.append(k)
        m = max(lw)
        w = [math.exp(v - m) for v in lw]
        t = sum(w)
        for s in range(Sn):
            W[s][i] = w[s] / t
    return RichResult(payload={
        "estimate": max(ks), "k": ks, "weights": W, "S": Sn, "n": n,
        "method": "Pareto-smoothed importance weights"})


def cheatsheet():
    return "loopr: Pareto-smoothed importance weights."
