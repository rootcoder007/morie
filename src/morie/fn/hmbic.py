# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian information criterion for cluster-number selection."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_bic"]


def geron_bic(log_lik, k, n):
    """
    Bayesian information criterion for cluster-number selection.

    Formula: BIC = -2 log L + k log(n)

    Parameters
    ----------
    log_lik : float or array-like
        Maximised log-likelihood(s).
    k : int or array-like
        Number of free parameters (non-negative integer).
    n : int
        Sample size; must be at least 1 so that log(n) is defined.

    Returns
    -------
    result : RichResult
        Keys: bic, delta, weights, best_index, estimate, n, method.

    Examples
    --------
    >>> r = geron_bic(-10.0, 2, 100)
    >>> round(float(r["bic"]), 5)
    29.21034
    >>> r2 = geron_bic([-10.0, -8.0], [2, 6], 100)
    >>> [round(float(v), 5) for v in r2["bic"]]
    [29.21034, 43.63102]
    >>> r2["best_index"]
    0

    References
    ----------
    Géron Ch 8
    """
    ll = np.atleast_1d(np.asarray(log_lik, dtype=float))
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    if ll.size == 0:
        raise ValueError("geron_bic: log_lik is empty")
    if kk.size == 1 and ll.size > 1:
        kk = np.repeat(kk, ll.size)
    if ll.size == 1 and kk.size > 1:
        ll = np.repeat(ll, kk.size)
    if ll.shape != kk.shape:
        raise ValueError(f"geron_bic: log_lik shape {ll.shape} does not match k shape {kk.shape}")
    if not np.all(np.isfinite(ll)):
        raise ValueError("geron_bic: log_lik contains non-finite values")
    if np.any(kk < 0) or np.any(kk != np.floor(kk)):
        raise ValueError("geron_bic: k must be a non-negative integer count of free parameters")
    nn = float(n)
    if nn < 1:
        raise ValueError("geron_bic: n must be >= 1")

    bic = -2.0 * ll + kk * np.log(nn)
    delta = bic - float(np.min(bic))
    w = np.exp(-0.5 * delta)
    weights = w / float(np.sum(w))
    best = int(np.argmin(bic))

    scalar = ll.size == 1
    return RichResult(
        title="Bayesian information criterion",
        summary_lines=[("BIC", float(bic[best])), ("Best model index", best)],
        payload={
            "bic": float(bic[0]) if scalar else bic,
            "delta": float(delta[0]) if scalar else delta,
            "weights": float(weights[0]) if scalar else weights,
            "best_index": best,
            "penalty": float(np.log(nn)),
            "estimate": float(bic[best]),
            "n": int(nn),
            "method": "Bayesian information criterion (BIC = -2 log L + k log n)",
        },
    )


def cheatsheet():
    return "hmbic: Bayesian information criterion for cluster-number selection"
