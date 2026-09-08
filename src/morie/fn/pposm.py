# morie.fn -- function file (rootcoder007/morie)
"""Posterior predictive mean and spread from replicated draws."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["ppmean", "posterior_predictive_mean"]


def ppmean(yrep):
    """Posterior predictive mean, sd and interval from replicated datasets.

    The predictive spread is NOT the posterior spread of the mean: it
    carries both the parameter uncertainty and the sampling noise of a
    future observation, so it is always the wider of the two.  Both are
    returned -- ``sd`` across replicate means and ``sd_pooled`` across
    all replicated observations -- because reporting the first when the
    second was meant is the standard way predictive intervals come out
    far too narrow.

    Formula: E[ytilde | y] ~= (1/S) sum_s ytilde^s, with the interval
             taken as empirical quantiles of the pooled draws

    Parameters
    ----------
    yrep : array-like, shape (S, n)
        Row s is one replicated dataset drawn from the posterior
        predictive distribution.

    Returns
    -------
    RichResult
        ``estimate``, ``sd``, ``sd_pooled``, ``ci_lower``,
        ``ci_upper``, ``rep_mean``, ``S``, ``n``.

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 6.3, which draws ytilde from
    the posterior predictive distribution by drawing theta from the
    posterior and then ytilde from p(ytilde | theta), and summarises it
    by simulation.  Fetched as the full text of the book from the
    author's own copy.
    """
    Y = C.mat(yrep)
    S = len(Y)
    if S < 2:
        raise ValueError("at least two replicated datasets are required")
    n = len(Y[0])
    if any(len(r) != n for r in Y):
        raise ValueError("every replicate must have the same length")
    rm = [sum(r) / n for r in Y]
    est = sum(rm) / S
    pooled = [v for r in Y for v in r]
    q = sorted(pooled)
    N = len(q)
    lo = q[max(0, int(math.floor(0.025 * (N - 1))))]
    hi = q[min(N - 1, int(math.ceil(0.975 * (N - 1))))]
    return RichResult(payload={
        "estimate": est, "sd": C.sd(rm, 1), "sd_pooled": C.sd(pooled, 1),
        "ci_lower": lo, "ci_upper": hi, "rep_mean": rm, "S": float(S),
        "n": float(n),
        "method": "Posterior predictive summary, BDA3 Section 6.3"})


posterior_predictive_mean = ppmean


def cheatsheet():
    return "pposm: predictive sd (pooled) is wider than the sd of replicate means"
