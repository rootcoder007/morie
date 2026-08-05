# SPDX-License-Identifier: AGPL-3.0-or-later
"""Andersen-Gill model for recurrent events."""

from . import _array_core as np

from ._richresult import RichResult
from ._recur_core import cox_counting_process

__all__ = ["agrec", "andersen_gill_recurrent"]


def andersen_gill_recurrent(start, stop, event, X, max_iter=50, tol=1e-9):
    """
    Andersen-Gill counting-process Cox model for recurrent events.

    Intensity: lambda_i(t | H_i) = Y_i(t) lambda_0(t) exp(beta' X_i),
    with Y_i(t) = 1{start_i < t <= stop_i}. Each subject contributes one
    row per at-risk interval; beta solves the Breslow partial-likelihood
    score over counting-process risk sets. With one interval per subject
    starting at zero this reduces exactly to the ordinary Cox model.

    Reference: Andersen & Gill (1982), Annals of Statistics 10(4),
    1100-1120, "Cox's regression model for counting processes".

    Parameters
    ----------
    start, stop : array-like
        Left-open interval endpoints, stop > start.
    event : array-like
        1 if an event ends the interval, 0 if censored.
    X : array-like
        Covariate matrix, one row per interval.

    Returns
    -------
    result : RichResult
        Keys: estimate (beta), se, cov, loglik, n_iter, n_events.
    """
    fit = cox_counting_process(start, stop, event, X,
                               max_iter=max_iter, tol=tol)
    return RichResult(payload={
        "estimate": fit["beta"],
        "se": fit["se"],
        "cov": fit["cov"],
        "loglik": fit["loglik"],
        "n_iter": fit["n_iter"],
        "n_events": fit["n_events"],
        "method": "Andersen-Gill (1982) counting-process Cox, Breslow ties",
    })


def cheatsheet():
    return "andersen_gill_recurrent(start, stop, event, X) -> counting-process Cox for recurrent events."


agrec = andersen_gill_recurrent
