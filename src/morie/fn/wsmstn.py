# morie.fn -- function file (rootcoder007/morie)
"""Sufficient statistics for the Bernoulli and Normal models."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["suffstat", "wasserman_sufficient"]


def suffstat(x, family="normal"):
    """Sufficient statistics, with the factorisation made explicit.

    Sufficiency is a claim about the LIKELIHOOD, not about the data, so
    what is returned alongside the statistics is the log-likelihood
    written in terms of them: the part that depends on the parameter
    reaches the data only through T.  Sufficient statistics are far
    from unique -- (17 Xbar, S) is sufficient too -- so this returns
    the standard minimal choice.

    Formula: Bernoulli   T = sum_i x_i
             Normal      T = (xbar, s), with
                         log f = -n log sigma - n s^2_ml/(2 sigma^2)
                                 - n (xbar - mu)^2/(2 sigma^2) + const

    Parameters
    ----------
    x : array-like
        The sample.
    family : {"bernoulli", "normal"}
        Which model the statistic is sufficient for.

    Returns
    -------
    RichResult
        ``T1``, ``T2`` (nan for the Bernoulli), ``n``, ``dim``, and
        for the Normal also ``mle_mu``, ``mle_sigma2``.

    References
    ----------
    Wasserman (2004), All of Statistics, Section 9.13.2, Definition
    9.32 and Examples 9.33 and 9.34: for Bernoulli(p),
    L(p) = p^S (1-p)^(n-S) with S = sum_i X_i, so S is sufficient; for
    N(mu, sigma), T = (Xbar, S) is sufficient because the density
    depends on the data only through T.  Fetched as the full text of
    the book.
    """
    x = C.vec(x)
    n = len(x)
    if n < 1:
        raise ValueError("the sample must be non-empty")
    fam = str(family).lower()
    if fam == "bernoulli":
        if any(v not in (0.0, 1.0) for v in x):
            raise ValueError("Bernoulli data must be 0/1")
        S = sum(x)
        return RichResult(payload={
            "T1": S, "T2": float("nan"), "n": n, "dim": 1.0,
            "mle_mu": S / n, "mle_sigma2": float("nan"),
            "method": "Bernoulli sufficient statistic, Wasserman Ex 9.33"})
    if fam == "normal":
        if n < 2:
            raise ValueError("the Normal statistic needs at least two points")
        m = sum(x) / n
        s = C.sd(x, 1)
        return RichResult(payload={
            "T1": m, "T2": s, "n": n, "dim": 2.0, "mle_mu": m,
            "mle_sigma2": sum((v - m) ** 2 for v in x) / n,
            "method": "Normal sufficient statistic, Wasserman Ex 9.34"})
    raise ValueError("family must be 'bernoulli' or 'normal'")


wasserman_sufficient = suffstat


def cheatsheet():
    return "wsmstn: Bernoulli T = sum x; Normal T = (xbar, s)"
