# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian information criterion."""

import math

from ._richresult import RichResult

__all__ = ["wasserman_bic"]


def wasserman_bic(loglik, k, n):
    """
    BIC of a fitted model.

    Formula: BIC = -2 log L_hat + k log n. Lower is better. The
    payload carries Wasserman's Ch 13 form
    BIC_W = log L_hat - (k/2) log n (higher is better) and the
    Schwarz-weight-ready difference is left to the caller. n must
    exceed 1 or log n gives the penalty the wrong sign.

    Parameters
    ----------
    loglik : float
        Maximised log-likelihood.
    k : int
        Number of estimated parameters, >= 0.
    n : int
        Sample size, >= 2.

    Returns
    -------
    result : dict
        Keys: estimate (classical BIC), bic_wasserman, loglik, k, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 13, section 13.6; Schwarz (1978).

    Examples
    --------
    >>> import math
    >>> out = wasserman_bic(-100.0, 3, 50)
    >>> round(out["estimate"], 12) == round(200.0 + 3 * math.log(50), 12)
    True
    >>> round(out["bic_wasserman"], 12) == round(-100.0 - 1.5 * math.log(50), 12)
    True
    >>> wasserman_bic(-100.0, 3, 1)
    Traceback (most recent call last):
        ...
    ValueError: BIC needs n >= 2; got 1.
    """
    loglik = float(loglik)
    k = int(k)
    n = int(n)
    if k < 0:
        raise ValueError(f"the parameter count cannot be negative; got {k}.")
    if n < 2:
        raise ValueError(f"BIC needs n >= 2; got {n}.")
    return RichResult(payload={
        "estimate": float(-2.0 * loglik + k * math.log(n)),
        "bic_wasserman": float(loglik - 0.5 * k * math.log(n)),
        "loglik": loglik, "k": k, "n": n,
        "method": "BIC = -2 log L + k log n; Wasserman form alongside"})


def cheatsheet():
    return "wsmbic: -2 ll + k log n; ll - (k/2) log n variant in payload"


# compact alias per ledger/NAMING.md
wassermanbic = wasserman_bic
