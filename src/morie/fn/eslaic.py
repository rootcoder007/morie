# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AIC model-selection score (ESL Ch 7.5)."""

from ._richresult import RichResult

__all__ = ["esl_aic_score"]


def esl_aic_score(loglik, d):
    """
    Akaike information criterion.

    Formula: AIC = -2 log L + 2 d, with d the effective number of
    parameters. Lower is better. ESL Eq. 7.29 states the same
    criterion per observation and in the Gaussian case as
    AIC = -(2/N) loglik + 2 (d/N) sigma^2; that scaling changes the
    units but not the ranking, so the classical form is the estimate
    and the per-observation variant is left to wsmaic's caller.

    Parameters
    ----------
    loglik : float
        Maximised log-likelihood.
    d : int
        Effective number of parameters, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (AIC), loglik, d, method.

    References
    ----------
    Hastie, Tibshirani and Friedman (2009), *The Elements of
    Statistical Learning*, 2nd ed., Ch 7.5 (Eq. 7.29).

    Examples
    --------
    >>> esl_aic_score(-100.0, 3)["estimate"]
    206.0
    >>> esl_aic_score(-50.0, 0)["estimate"]
    100.0
    >>> esl_aic_score(-100.0, -1)
    Traceback (most recent call last):
        ...
    ValueError: the parameter count cannot be negative; got -1.
    """
    loglik = float(loglik)
    d = int(d)
    if d < 0:
        raise ValueError(f"the parameter count cannot be negative; got {d}.")
    return RichResult(payload={
        "estimate": -2.0 * loglik + 2.0 * d, "loglik": loglik, "d": d,
        "method": "AIC = -2 log L + 2 d"})


def cheatsheet():
    return "eslaic: AIC = -2 log L + 2d; lower is better"


# compact alias per ledger/NAMING.md
eslaicscore = esl_aic_score
