# morie.fn -- Hedderich shelf, rebuilt from the book (rootcoder007/morie)
"""Residual analysis for the logistic regression model.

Source READ FROM THE CORPUS PDF, pages rendered with ``pdftoppm``:
Hedderich, Sachs and Reynarowych, *Applied Statistics: Methods Using R*,
section 8.4.5 "Residual Analysis", printed pages 837-838, equations
(8.67), (8.68) and (8.69).

Pearson residuals (8.67)::

    r_i = (y_i - n_i pihat_i) / sqrt(n_i pihat_i (1 - pihat_i))

Model deviance and deviance residuals (8.68)::

    D = sum_i d_i^2
    d_i = +/- sqrt( 2 [ y_i log(y_i / (n_i pihat_i))
                        + (n_i - y_i) log((n_i - y_i) / (n_i (1 - pihat_i))) ] )

BOOK ERRATUM in (8.68): the printed radicand is
``-2(y log(y/(n pihat))) + (n - y) log(...)``.  Read literally the
leading factor is -2 on the first term only, which makes the radicand
negative for a well-fitting observation and the square root imaginary.
The bracket is the Kullback-Leibler divergence of the saturated model
from the fitted one and is non-negative, so the factor is +2 over the
whole bracket -- the standard published binomial deviance residual, and
the one that base R returns from residuals(fit, "deviance").  Verified against
that route; see the anchors.  The sign is ``sign(y_i - n_i pihat_i)``,
which the book writes as the leading ``+/-``.

Influence measure (8.69)::

    Delta D_i = d_i^2 + r_i^2 h_ii / (1 - h_ii)

with h_ii the hat-matrix diagonal from the IRLS solution.  The book
gives this same line as R code on page 838::

    idev <- deviance.resid^2 + pearson.resid^2 * hats/(1-hats)

The terms ``0 log 0`` arising when ``y_i = 0`` or ``y_i = n_i`` are
taken as 0, their limiting value.
"""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["lrresid"]


def _xlogxy(x, y):
    """x * log(x / y) with the limit 0 at x = 0."""
    if x == 0.0:
        return 0.0
    return x * math.log(x / y)


def lrresid(y, pihat, n=None, hat=None):
    """Pearson and deviance residuals, model deviance, influence.

    Parameters
    ----------
    y : array-like
        Observed counts of successes.  For ungrouped 0/1 data these are
        the 0/1 responses and ``n`` is all ones.
    pihat : array-like
        Fitted probabilities, strictly inside (0, 1).
    n : array-like, optional
        Binomial denominators; defaults to all ones.
    hat : array-like, optional
        Hat-matrix diagonal.  If given, ``delta_d`` (8.69) is returned.

    Returns
    -------
    RichResult
        Keys: ``pearson``, ``deviance``, ``D``, ``n``, and when ``hat``
        is given ``hat`` and ``delta_d``.
    """
    y = [float(v) for v in np.asarray(y, dtype=float).ravel()]
    p = [float(v) for v in np.asarray(pihat, dtype=float).ravel()]
    m = len(y)
    if m == 0:
        raise ValueError("y must not be empty")
    if len(p) != m:
        raise ValueError("y and pihat must have the same length")
    if n is None:
        nn = [1.0] * m
    else:
        nn = [float(v) for v in np.asarray(n, dtype=float).ravel()]
        if len(nn) != m:
            raise ValueError("y and n must have the same length")
    for v in p:
        if not (0.0 < v < 1.0):
            raise ValueError("every fitted probability must lie strictly inside (0, 1)")
    for i in range(m):
        if nn[i] <= 0.0:
            raise ValueError("every binomial denominator must be positive")
        if not (0.0 <= y[i] <= nn[i]):
            raise ValueError("every y must satisfy 0 <= y <= n")
    pear = []
    dev = []
    for i in range(m):
        mu = nn[i] * p[i]
        pear.append((y[i] - mu) / math.sqrt(nn[i] * p[i] * (1.0 - p[i])))
        bracket = _xlogxy(y[i], mu) + _xlogxy(nn[i] - y[i], nn[i] * (1.0 - p[i]))
        s = 2.0 * bracket
        if s < 0.0:
            s = 0.0
        d = math.sqrt(s)
        dev.append(d if y[i] >= mu else -d)
    D = 0.0
    for d in dev:
        D += d * d
    payload = {"pearson": pear, "deviance": dev, "D": D, "n": nn}
    summary = [("Model deviance D", D), ("observations", m)]
    if hat is not None:
        h = [float(v) for v in np.asarray(hat, dtype=float).ravel()]
        if len(h) != m:
            raise ValueError("y and hat must have the same length")
        for v in h:
            if not (0.0 <= v < 1.0):
                raise ValueError("every hat value must lie in [0, 1)")
        payload["hat"] = h
        payload["delta_d"] = [dev[i] ** 2 + pear[i] ** 2 * h[i] / (1.0 - h[i]) for i in range(m)]
    return RichResult(
        title="Logistic regression residual analysis (Hedderich eqs. 8.67-8.69)",
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet() -> str:
    return "lrresid(y, pihat, n, hat): Pearson/deviance residuals, D, influence -- Hedderich eqs. (8.67)-(8.69)."
