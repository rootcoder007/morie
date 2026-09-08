# morie.fn -- function file (rootcoder007/morie)
"""Pareto k importance-weight diagnostic."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pareto_k_diagnostic"]


def pareto_k_diagnostic(log_lik):
    """Shape of the importance-weight tail, per observation.

    The estimate is only as good as the importance weights, and their
    variance is finite only when the tail shape is below one half.  So
    ``k`` is not a fit statistic, it is a statement about whether the
    computation is entitled to a central limit theorem at all: past 0.7
    the sample size needed grows so fast that the estimate should not be
    used, and the right response is to refit that fold exactly.

    Formula: fit a generalised Pareto to the largest
    ``M = min(0.2 S, 3 sqrt(S))`` weights by the Zhang-Stephens
    empirical-Bayes rule; ``k`` is its shape.

    Parameters
    ----------
    log_lik : array-like, shape (S, n)
        Pointwise log likelihood.

    Returns
    -------
    RichResult
        ``estimate`` (largest k), ``k``, ``n_bad`` (k above 0.7),
        ``n_ok``, ``S``, ``n``.

    References
    ----------
    Vehtari, A., Simpson, D., Gelman, A., Yao, Y. & Gabry, J. (2024).
    Pareto smoothed importance sampling.  Journal of Machine Learning
    Research 25:1-58.  The generalised Pareto fit is Zhang, J. &
    Stephens, M. A. (2009), A new and efficient estimation method for
    the generalized Pareto distribution, Technometrics 51:316-325.
    """
    L = C.mat(log_lik)
    Sn = len(L)
    n = len(L[0])
    ks = []
    for i in range(n):
        _, k = S.psis([-L[s][i] for s in range(Sn)])
        ks.append(k)
    bad = sum(1 for v in ks if v > 0.7)
    return RichResult(payload={
        "estimate": float("nan") if any(v != v for v in ks) else max(ks), "k": ks, "n_bad": bad, "n_ok": n - bad,
        "S": Sn, "n": n, "method": "Pareto k importance-weight diagnostic"})


def cheatsheet():
    return "khatd: Pareto k importance-weight diagnostic."
