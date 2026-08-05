# morie.fn -- function file (rootcoder007/morie)
"""Gaussian conditional log-likelihood.

Hastie, Tibshirani and Friedman (2009), *The Elements of Statistical
Learning*, 2nd ed., Springer, Section 2.6.3, book p. 31 (PDF p. 50):

    L(theta) = sum_i log Pr_theta(y_i)                              (2.33)
    Pr(Y|X, theta) = N(f_theta(X), sigma^2)                         (2.34)
    L(theta) = -(N/2) log(2 pi) - N log sigma
               - (1/(2 sigma^2)) sum_i (y_i - f_theta(x_i))^2       (2.35)

Note the middle term is -N log sigma, not -(N/2) log sigma^2 written
carelessly: they are equal, and (2.35) is coded in the book's form.
When sigma is not supplied it is profiled out at its maximum-likelihood
value sqrt(RSS/N), which is what makes least squares and maximum
likelihood coincide here.
"""

from __future__ import annotations

import math

from . import _s03core as k

from ._richresult import RichResult

__all__ = ["gausslik"]


def gausslik(y, mu, sigma=None):
    """Equation (2.35).

    Parameters
    ----------
    y : array-like
        N-vector of observations.
    mu : array-like or float
        Fitted values f_theta(x_i); a scalar is recycled.
    sigma : float, optional
        Standard deviation; profiled to sqrt(RSS/N) when omitted.

    Returns
    -------
    RichResult with keys estimate, loglik, rss, sigma, n, aic, method.
    """
    yv = k.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("gausslik: y is empty")
    try:
        mv = [float(mu)] * n
    except (TypeError, ValueError):
        mv = k.vec(mu)
    if len(mv) != n:
        raise ValueError("gausslik: y and mu must have the same length")
    rss = sum((yv[i] - mv[i]) ** 2 for i in range(n))
    if sigma is None:
        if rss <= 0.0:
            raise ValueError("gausslik: cannot profile sigma from a zero residual sum of squares")
        s = math.sqrt(rss / n)
    else:
        s = float(sigma)
        if s <= 0.0:
            raise ValueError("gausslik: sigma must be positive")
    ll = -0.5 * n * math.log(2.0 * math.pi) - n * math.log(s) - rss / (2.0 * s * s)
    return RichResult(
        title="Gaussian log-likelihood, ESL eq. (2.35)",
        summary_lines=[("n", n), ("sigma", s), ("loglik", ll)],
        payload={
            "estimate": ll,
            "loglik": ll,
            "rss": rss,
            "sigma": s,
            "n": n,
            "aic": -2.0 * ll + 2.0,
            "method": "Hastie-Tibshirani-Friedman (2009) ESL eqs. (2.33)-(2.35)",
        },
    )


def cheatsheet():
    return "gausslik: Gaussian conditional log-likelihood, ESL eqs. (2.34)-(2.35)"
