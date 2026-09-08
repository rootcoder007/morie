# morie.fn -- function file (rootcoder007/morie)
"""Gamma-Poisson posterior and predictive distribution."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["poispred", "poisson_predictive"]


def poispred(y, alpha, beta, exposure=None):
    """Conjugate Poisson analysis with its negative-binomial predictive.

    The predictive is NEGATIVE BINOMIAL, not Poisson: integrating the
    rate out of a Poisson against its Gamma posterior adds
    overdispersion, so the predictive variance strictly exceeds the
    predictive mean.  Plugging the posterior mean into a Poisson
    instead understates the spread by exactly the factor returned as
    ``overdispersion``.

    Formula: theta | y ~ Gamma(alpha + sum y_i, beta + sum e_i);
             ytilde | y ~ NegBin(alpha', beta'/(beta' + etilde)) with
             mean etilde alpha'/beta' and
             variance etilde alpha'/beta' (1 + etilde/beta')

    Parameters
    ----------
    y : array-like
        Observed counts, non-negative.
    alpha : float
        Prior shape, alpha > 0.
    beta : float
        Prior rate, beta > 0.
    exposure : array-like, optional
        Exposure e_i for each observation (default: all 1).  The
        predictive is for one further unit of exposure.

    Returns
    -------
    RichResult
        ``alpha_post``, ``beta_post``, ``rate_mean``, ``rate_var``,
        ``pred_mean``, ``pred_var``, ``overdispersion``, ``n``.

    References
    ----------
    Gelman, Carlin, Stern, Dunson, Vehtari & Rubin (2013), Bayesian
    Data Analysis, 3rd edition, Section 2.6 (the Poisson model with its
    conjugate Gamma prior) and Section 2.7 (the negative-binomial
    predictive obtained by integrating the Gamma out).  Fetched as the
    full text of the book from the author's own copy.
    """
    y = C.vec(y)
    n = len(y)
    if n < 1:
        raise ValueError("at least one observation is required")
    if any(v < 0 for v in y):
        raise ValueError("counts must be non-negative")
    a = float(alpha)
    b = float(beta)
    if a <= 0 or b <= 0:
        raise ValueError("the Gamma prior needs alpha > 0 and beta > 0")
    if exposure is None:
        e = [1.0] * n
    else:
        e = C.vec(exposure)
        if len(e) != n:
            raise ValueError("exposure must have one entry per observation")
        if any(v <= 0 for v in e):
            raise ValueError("exposures must be positive")
    ap = a + sum(y)
    bp = b + sum(e)
    pm = ap / bp
    pv = pm * (1.0 + 1.0 / bp)
    return RichResult(payload={
        "alpha_post": ap, "beta_post": bp, "rate_mean": pm,
        "rate_var": ap / (bp * bp), "pred_mean": pm, "pred_var": pv,
        "overdispersion": pv / pm, "n": float(n),
        "method": "Gamma-Poisson posterior and negative-binomial predictive"})


poisson_predictive = poispred


def cheatsheet():
    return "poispr: theta|y ~ Ga(a+sum y, b+sum e); ytilde ~ NegBin, var > mean"
