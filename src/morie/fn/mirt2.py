# morie.fn -- function file (rootcoder007/morie)
"""Compensatory multidimensional IRT (M3PL / M2PL)."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mirt_2d_compensatory", "mirt2dcompensatory"]


def mirt_2d_compensatory(y, theta, a, d, c=0.0, D=1.0):
    """Compensatory multidimensional item response probabilities.

    Verified against Chalmers (2012), JSS 48(6), p.3 eq. (1), read from
    a rendered image of the page rather than the text layer:

        Phi(x = 1 | theta, alpha, d, gamma)
            = gamma + (1 - gamma) / (1 + exp[-D (alpha' theta + d)])

    "Compensatory" is the content of ``alpha' theta``: the abilities
    enter as a single weighted sum, so a low ``theta_1`` can be repaid by
    a high ``theta_2``.  A noncompensatory model multiplies per-dimension
    probabilities instead and admits no such trade.

    ``D`` is a scaling adjustment, 1 for the logistic metric and 1.702
    for the normal-ogive metric; the default here is 1, so the returned
    probability is the plain logistic one.

    The stub this replaces printed the exponent as
    ``-(a1 theta1 + a2 theta2) + d``, which puts ``d`` outside the
    negation and inverts the role of the intercept.  Equation (1) has
    ``-D(alpha' theta + d)``: the whole linear predictor is negated.  The
    published form is what is implemented.

    Nothing restricts this to two dimensions -- ``a`` sets the
    dimension, and ``mirt3`` is this same function with three columns
    and no guessing.

    Parameters
    ----------
    y : array-like of 0/1, length n
        Observed responses, used for the Bernoulli log-likelihood.
    theta : array-like, shape (n, m)
        Ability vector per respondent.  A flat length-n sequence is read
        as ``m = 1``.
    a : array-like, length m
        Item slopes (discriminations), one per dimension.
    d : float
        Item intercept.
    c : float, default 0
        Lower asymptote (guessing).  ``c = 0`` gives the M2PL.
    D : float, default 1
        Metric constant; 1.702 converts to the normal-ogive metric.

    Returns
    -------
    RichResult
        ``estimate`` (log-likelihood), ``loglik``, ``p`` (probability
        per respondent), ``pbar``, ``deviance``, ``n``, ``m``.

    References
    ----------
    Chalmers, R. P. (2012), "mirt: A Multidimensional Item Response
    Theory Package for the R Environment", Journal of Statistical
    Software 48(6), 1-29, doi:10.18637/jss.v048.i06, eq. (1) p.3.
    Reckase, M. D. (2009), Multidimensional Item Response Theory,
    Springer, which Chalmers cites for the metric constant D; the book
    itself was not in the local corpus and was not consulted.
    """
    av = C.vec(a)
    m = len(av)
    if m == 0:
        raise ValueError("a must name at least one dimension")
    yv = C.vec(y)
    n = len(yv)
    if n == 0:
        raise ValueError("y is empty")
    if any(v != 0.0 and v != 1.0 for v in yv):
        raise ValueError("y must be binary 0/1")
    cc = float(c)
    if not (0.0 <= cc < 1.0):
        raise ValueError("c must lie in [0, 1)")
    dd = float(d)
    Dm = float(D)
    if m == 1:
        th = [[v] for v in C.vec(theta)]
    else:
        th = C.mat(theta)
    if len(th) != n:
        raise ValueError("theta must have one row per response")
    for row in th:
        if len(row) != m:
            raise ValueError("theta must have one column per element of a")
    p = []
    ll = 0.0
    for i in range(n):
        z = dd
        for k in range(m):
            z += av[k] * th[i][k]
        pi = cc + (1.0 - cc) / (1.0 + math.exp(-Dm * z))
        p.append(pi)
        ll += math.log(pi) if yv[i] == 1.0 else math.log(1.0 - pi)
    pbar = sum(p) / n
    return RichResult(payload={
        "estimate": ll, "loglik": ll, "p": p, "pbar": pbar,
        "deviance": -2.0 * ll, "n": n, "m": m,
        "method": "Compensatory multidimensional IRT (Chalmers 2012 eq. 1)"})


mirt2dcompensatory = mirt_2d_compensatory


def cheatsheet():
    return "mirt2: Compensatory multidimensional IRT item response probabilities"
