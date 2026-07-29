# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""BIC model-selection score (ESL Ch 7.7)."""

import math

from ._richresult import RichResult

__all__ = ["esl_bic_score"]


def esl_bic_score(loglik, d, N):
    """
    Bayesian information criterion.

    Formula: BIC = -2 log L + d log N. Lower is better. BIC penalises
    complexity more heavily than AIC once N > e^2 ~ 7.389, so the
    payload reports which of the two penalties is larger here rather
    than leaving the comparison implicit.

    Parameters
    ----------
    loglik : float
        Maximised log-likelihood.
    d : int
        Effective number of parameters, >= 0.
    N : int
        Sample size, >= 2 (log N must be positive for a penalty).

    Returns
    -------
    result : dict
        Keys: estimate (BIC), penalty, aic_penalty,
        penalises_more_than_aic, loglik, d, N, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), Ch 7.7 (Eq. 7.35).

    Examples
    --------
    >>> import math
    >>> out = esl_bic_score(-100.0, 3, 50)
    >>> round(out["estimate"], 12) == round(200.0 + 3 * math.log(50), 12)
    True
    >>> out["penalises_more_than_aic"]
    True
    >>> esl_bic_score(-100.0, 3, 5)["penalises_more_than_aic"]
    False
    >>> esl_bic_score(-100.0, 3, 1)
    Traceback (most recent call last):
        ...
    ValueError: BIC needs N >= 2; got 1.
    """
    loglik = float(loglik)
    d = int(d)
    N = int(N)
    if d < 0:
        raise ValueError(f"the parameter count cannot be negative; got {d}.")
    if N < 2:
        raise ValueError(f"BIC needs N >= 2; got {N}.")
    penalty = d * math.log(N)
    return RichResult(payload={
        "estimate": -2.0 * loglik + penalty, "penalty": penalty,
        "aic_penalty": 2.0 * d,
        "penalises_more_than_aic": bool(penalty > 2.0 * d),
        "loglik": loglik, "d": d, "N": N,
        "method": "BIC = -2 log L + d log N"})


def cheatsheet():
    return "eslbic: BIC = -2 log L + d log N; harsher than AIC once N > e^2"
