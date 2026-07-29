# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mallows' C_p statistic (ESL Ch 7.5)."""

from ._richresult import RichResult

__all__ = ["esl_mallows_cp"]


def esl_mallows_cp(RSS, d, n, sigma2):
    """
    Mallows' C_p, in ESL's in-sample-error scaling.

    Formula: C_p = (1/n)(RSS + 2 d sigma^2), an estimate of in-sample
    prediction error; sigma^2 is the noise variance estimated from a
    LOW-BIAS model, not from the candidate itself (using the
    candidate's own residual variance makes the criterion circular).
    The classical Mallows form C_p = RSS/sigma^2 - n + 2d ships
    alongside as ``cp_classical``: the two rank models identically but
    differ in scale and sign conventions, and mixing them up is the
    usual error.

    Parameters
    ----------
    RSS : float
        Residual sum of squares of the candidate model, >= 0.
    d : int
        Number of parameters, >= 0.
    n : int
        Sample size, >= 1.
    sigma2 : float
        Noise variance from a low-bias model, > 0.

    Returns
    -------
    result : dict
        Keys: estimate (ESL scaling), cp_classical, RSS, d, n,
        sigma2, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 7.5 (Eq. 7.26);
    Mallows (1973).

    Examples
    --------
    >>> out = esl_mallows_cp(100.0, 3, 50, 2.0)
    >>> out["estimate"]
    2.24
    >>> out["cp_classical"]
    6.0
    >>> esl_mallows_cp(100.0, 3, 50, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: the noise variance must be positive; got 0.0.
    """
    RSS = float(RSS)
    d = int(d)
    n = int(n)
    sigma2 = float(sigma2)
    if RSS < 0:
        raise ValueError(f"the residual sum of squares cannot be negative; got {RSS}.")
    if d < 0:
        raise ValueError(f"the parameter count cannot be negative; got {d}.")
    if n < 1:
        raise ValueError(f"C_p needs n >= 1; got {n}.")
    if sigma2 <= 0:
        raise ValueError(f"the noise variance must be positive; got {sigma2}.")
    return RichResult(payload={
        "estimate": (RSS + 2.0 * d * sigma2) / n,
        "cp_classical": RSS / sigma2 - n + 2.0 * d,
        "RSS": RSS, "d": d, "n": n, "sigma2": sigma2,
        "method": "C_p = (1/n)(RSS + 2 d sigma^2), ESL scaling"})


def cheatsheet():
    return "eslcp: C_p = (RSS + 2 d sigma^2)/n; classical form alongside"
