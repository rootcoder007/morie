# morie.fn -- slice s03 (rootcoder007/morie)
"""Schwarz's Bayesian information criterion.

Source consulted: Schwarz, G. (1978).  Estimating the dimension of a
model.  *The Annals of Statistics* 6(2), 461-464.  Schwarz derives the
criterion as the leading terms of the Laplace approximation to the log
marginal likelihood, and states it in the form

    log M_j = log L_j - (k_j / 2) log n + O(1)

which, written as a deviance to be *minimised*, is the familiar

    BIC = -2 log L + p log n

with p the number of estimated parameters and n the sample size.  The
1978 Annals paper is behind a paywall, so the equation above was taken
from its standard published form rather than from the PDF itself; the
form is unambiguous and identical in every secondary source.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401  (kept for module conventions)

from ._richresult import RichResult

__all__ = ["bayesian_information_criterion"]


def bayesian_information_criterion(log_lik, n_params, n_obs):
    """Schwarz's BIC, and the companion AIC, for a fitted model.

    Parameters
    ----------
    log_lik : float
        Maximised log-likelihood of the model.
    n_params : int
        Number of freely estimated parameters, p.
    n_obs : int
        Number of observations, n.

    Returns
    -------
    RichResult with payload:
        estimate : BIC = -2 log L + p log n
        aic      : -2 log L + 2 p, for comparison
        penalty  : p log n
        log_lik, n_params, n
    """
    ll = float(log_lik)
    p = float(n_params)
    n = float(n_obs)
    penalty = p * math.log(n) if n > 0.0 else float("nan")
    bic = -2.0 * ll + penalty
    aic = -2.0 * ll + 2.0 * p
    return RichResult(
        title="Bayesian information criterion",
        summary_lines=[("BIC", bic), ("AIC", aic)],
        payload={
            "estimate": bic,
            "aic": aic,
            "penalty": penalty,
            "log_lik": ll,
            "n_params": p,
            "n": n,
            "method": "Schwarz (1978) Bayesian information criterion",
        },
    )


def cheatsheet():
    return "bicg: Bayesian information criterion (BIC)"
