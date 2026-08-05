# SPDX-License-Identifier: AGPL-3.0-or-later
"""WLW marginal Cox model for recurrent events."""

from . import _array_core as np

from ._richresult import RichResult
from ._recur_core import cox_counting_process

__all__ = ["wlwmm", "wlw_marginal_model"]


def wlw_marginal_model(time, event, X, occurrence, max_iter=50, tol=1e-9):
    """
    Wei-Lin-Weissfeld marginal model for multivariate failure times.

    Each occurrence k gets its own marginal Cox model on the TOTAL time
    scale (every subject is at risk for every occurrence from time 0):
    lambda_k(t) = lambda_0k(t) exp(beta_k' X). This function reports the
    per-occurrence marginal estimates and, as the common-effect summary,
    the stratified fit with a single beta across occurrence strata --
    the constrained estimator of WLW Section 3.

    Reference: Wei, Lin & Weissfeld (1989), JASA 84(408), 1065-1073,
    "Regression analysis of multivariate incomplete failure time data
    by modeling marginal distributions".

    Returns
    -------
    result : RichResult
        Keys: estimate (common beta), se, per_event_beta (dict
        occurrence -> beta vector), loglik, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape((-1, 1))
    if Xa.shape[0] != time.shape[0]:
        Xa = Xa.T
    occ = list(occurrence)
    zeros = np.zeros(time.shape[0])

    per_event = {}
    for k in sorted(set(occ)):
        idx = [i for i, o in enumerate(occ) if o == k]
        if not any(event[i] == 1.0 for i in idx):
            continue
        try:
            sub = cox_counting_process(
                np.asarray([0.0] * len(idx)),
                np.asarray([time[i] for i in idx]),
                np.asarray([event[i] for i in idx]),
                np.asarray([Xa[i].tolist() for i in idx]),
                max_iter=max_iter, tol=tol)
        except ValueError:
            continue
        per_event[k] = sub["beta"]

    fit = cox_counting_process(zeros, time, event, Xa, strata=occ,
                               max_iter=max_iter, tol=tol)
    return RichResult(payload={
        "estimate": fit["beta"],
        "se": fit["se"],
        "cov": fit["cov"],
        "per_event_beta": per_event,
        "loglik": fit["loglik"],
        "n_iter": fit["n_iter"],
        "n_events": fit["n_events"],
        "method": "Wei-Lin-Weissfeld (1989) marginal model, total-time stratified Cox, Breslow ties",
    })


def cheatsheet():
    return "wlw_marginal_model(time, event, X, occurrence) -> WLW marginal Cox model."


wlwmm = wlw_marginal_model
