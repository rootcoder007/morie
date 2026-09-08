# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Akaike information criterion."""

from ._richresult import RichResult

__all__ = ["wasserman_aic"]


def wasserman_aic(loglik, k):
    """
    AIC of a fitted model.

    Formula: AIC = -2 log L_hat + 2k, with k the number of estimated
    parameters. Lower is better. The payload also carries Wasserman's
    Ch 13 sign convention AIC_W = log L_hat - k (HIGHER is better)
    so both textbook forms are on hand; they rank models identically.

    Parameters
    ----------
    loglik : float
        Maximised log-likelihood.
    k : int
        Number of estimated parameters, >= 0.

    Returns
    -------
    result : dict
        Keys: estimate (classical AIC), aic_wasserman, loglik, k,
        method.

    References
    ----------
    Wasserman (2004), Ch 13, section 13.6; Akaike (1973).

    Examples
    --------
    >>> out = wasserman_aic(-100.0, 3)
    >>> out["estimate"]
    206.0
    >>> out["aic_wasserman"]
    -103.0
    >>> wasserman_aic(-100.0, -1)
    Traceback (most recent call last):
        ...
    ValueError: the parameter count cannot be negative; got -1.
    """
    loglik = float(loglik)
    k = int(k)
    if k < 0:
        raise ValueError(f"the parameter count cannot be negative; got {k}.")
    return RichResult(payload={
        "estimate": float(-2.0 * loglik + 2.0 * k),
        "aic_wasserman": float(loglik - k), "loglik": loglik, "k": k,
        "method": "AIC = -2 log L + 2k (classical); Wasserman form alongside"})


def cheatsheet():
    return "wsmaic: -2 ll + 2k; Wasserman's ll - k variant in payload"


# compact alias per ledger/NAMING.md
wassermanaic = wasserman_aic
