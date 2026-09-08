# morie.fn -- function file (rootcoder007/morie)
"""Log pointwise predictive density and WAIC."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["lppd", "log_pointwise_predictive_density"]


def lppd(logdens):
    """lppd, effective parameters and WAIC from a matrix of log densities.

    POINTWISE is the operative word: the average over draws happens
    INSIDE the log, once per observation, and only then are the
    observations summed.  Averaging the total log-likelihood over draws
    instead -- the obvious thing to write -- gives a different and
    wrong quantity.

    The computation is done through a log-sum-exp, not by
    exponentiating: with a few hundred observations the raw densities
    underflow to zero long before the lppd is large.

    Formula: lppd = sum_i log( (1/S) sum_s p(y_i | theta^s) );
             p_waic = sum_i var_s( log p(y_i | theta^s) );
             elpd_waic = lppd - p_waic;  WAIC = -2 elpd_waic

    Parameters
    ----------
    logdens : array-like, shape (S, n)
        Row s, column i holds log p(y_i | theta^s).

    Returns
    -------
    RichResult
        ``lppd``, ``p_waic``, ``elpd_waic``, ``waic``,
        ``pointwise_lppd``, ``pointwise_var``, ``S``, ``n``.

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 7.2, equation (7.5): the
    computed lppd is sum_{i=1}^{n} log( (1/S) sum_{s=1}^{S}
    p(y_i | theta^s) ), and the effective number of parameters
    p_waic is the sum over observations of the posterior variance of
    the log predictive density.  Fetched as the full text of the book
    from the author's own copy.
    """
    L = C.mat(logdens)
    S = len(L)
    if S < 2:
        raise ValueError("at least two posterior draws are required")
    n = len(L[0])
    if any(len(r) != n for r in L):
        raise ValueError("every draw must score every observation")
    pl = []
    pv = []
    for i in range(n):
        col = [L[s][i] for s in range(S)]
        m = max(col)
        pl.append(m + math.log(sum(math.exp(v - m) for v in col) / S))
        pv.append(C.var(col, 1))
    tot = sum(pl)
    pw = sum(pv)
    el = tot - pw
    return RichResult(payload={
        "lppd": tot, "p_waic": pw, "elpd_waic": el, "waic": -2.0 * el,
        "pointwise_lppd": pl, "pointwise_var": pv, "S": float(S),
        "n": float(n),
        "method": "lppd and WAIC, BDA3 Section 7.2 equation (7.5)"})


log_pointwise_predictive_density = lppd


def cheatsheet():
    return "lpdwc: lppd = sum_i log mean_s p(y_i|th^s); p_waic = sum_i var_s log p"
