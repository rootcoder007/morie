# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prentice-Williams-Peterson gap-time model."""

from . import _array_core as np

from ._richresult import RichResult
from ._recur_core import cox_counting_process

__all__ = ["pwpgt", "pwp_gap_time"]


def pwp_gap_time(start, stop, event, X, occurrence, max_iter=50, tol=1e-9):
    """
    Prentice-Williams-Peterson conditional gap-time model.

    Hazard for the k-th occurrence, on the gap-time scale:
    lambda_k(t | H) = lambda_0k(t - t_{k-1}) exp(beta' X). A subject is
    at risk for occurrence k only after occurrence k-1; risk sets are
    stratified by occurrence number and the clock restarts at each
    event, so the model is a stratified Cox fit on gap = stop - start
    with stratum = occurrence and a common beta across strata.

    Reference: Prentice, Williams & Peterson (1981), Biometrika 68(2),
    373-379, "On the regression analysis of multivariate failure time
    data".

    Returns
    -------
    result : RichResult
        Keys: estimate (beta), se, cov, loglik, n_iter, n_events.
    """
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    gap = stop - start
    zeros = np.zeros(gap.shape[0])
    fit = cox_counting_process(zeros, gap, event, X, strata=occurrence,
                               max_iter=max_iter, tol=tol)
    return RichResult(payload={
        "estimate": fit["beta"],
        "se": fit["se"],
        "cov": fit["cov"],
        "loglik": fit["loglik"],
        "n_iter": fit["n_iter"],
        "n_events": fit["n_events"],
        "method": "Prentice-Williams-Peterson (1981) gap-time stratified Cox, Breslow ties",
    })


def cheatsheet():
    return "pwp_gap_time(start, stop, event, X, occurrence) -> PWP conditional gap-time model."


pwpgt = pwp_gap_time
