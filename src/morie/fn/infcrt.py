# morie.fn -- function file (rootcoder007/morie)
"""WAIC alongside a PSIS-LOO estimate."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["information_criterion"]


def information_criterion(log_lik_samples):
    """WAIC and PSIS-LOO computed from the same draws, so they can disagree.

    They usually agree, and the informative case is when they do not:
    WAIC has no diagnostic of its own, while PSIS-LOO reports a Pareto
    shape for every observation and so can say WHICH point is breaking
    the approximation.  Both are returned, along with the largest shape,
    because a criterion without its diagnostic is a number with no error
    bar.

    Formula: ``WAIC = -2(lppd - p_WAIC)``; the LOO weights are
    ``w_is = 1 / exp(ll_is)`` smoothed by a generalised Pareto fitted to
    the largest ``min(0.2 S, 3 sqrt(S))`` of them.

    Parameters
    ----------
    log_lik_samples : array-like, shape (S, n)
        Pointwise log likelihood, draws by observations.

    Returns
    -------
    RichResult
        ``estimate`` (WAIC), ``looic``, ``elpd_loo``, ``p_loo``,
        ``k_max``, ``S``, ``n``.

    References
    ----------
    Watanabe, S. (2013).  A widely applicable Bayesian information
    criterion.  Journal of Machine Learning Research 14:867-897.  The
    PSIS-LOO half is Vehtari, A., Gelman, A. & Gabry, J. (2017),
    Practical Bayesian model evaluation using leave-one-out
    cross-validation and WAIC, Statistics and Computing 27:1413-1432.
    """
    L = C.mat(log_lik_samples)
    Sn = len(L)
    n = len(L[0])
    lppd = 0.0
    pw = 0.0
    elpd_loo = 0.0
    ks = []
    for i in range(n):
        col = [L[s][i] for s in range(Sn)]
        m = max(col)
        lppd += m + math.log(sum(math.exp(v - m) for v in col) / Sn)
        mu = sum(col) / Sn
        pw += sum((t - mu) ** 2 for t in col) / (Sn - 1)
        lw = [-v for v in col]
        sm, k = S.psis(lw)
        ks.append(k)
        mm = max(sm)
        num = math.log(sum(math.exp(sm[s] - mm + col[s]) for s in range(Sn)))
        den = math.log(sum(math.exp(sm[s] - mm) for s in range(Sn)))
        elpd_loo += num - den
    p_loo = lppd - elpd_loo
    return RichResult(payload={
        "estimate": -2.0 * (lppd - pw), "looic": -2.0 * elpd_loo,
        "elpd_loo": elpd_loo, "p_loo": p_loo, "k_max": max(ks), "S": Sn, "n": n,
        "method": "WAIC with a PSIS-LOO cross-check"})


def cheatsheet():
    return "infcrt: WAIC alongside a PSIS-LOO estimate."
