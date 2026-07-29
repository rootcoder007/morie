# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Akaike information criterion for cluster-number selection."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_aic"]


def geron_aic(log_lik, k, n=None):
    """
    Akaike information criterion for cluster-number selection.

    Formula: AIC = -2 log L + 2k

    Parameters
    ----------
    log_lik : float or array-like
        Maximised log-likelihood of the fitted model. May be a vector of
        log-likelihoods (one per candidate model) evaluated jointly with
        a matching vector of `k`.
    k : int or array-like
        Number of free parameters. Must be a non-negative integer.
    n : int, optional
        Sample size. When given, the small-sample corrected AICc is also
        reported; AICc is undefined when ``n - k - 1 <= 0``.

    Returns
    -------
    result : RichResult
        Keys: aic, aicc, delta, weights, estimate, n, method.

    Examples
    --------
    >>> r = geron_aic(-10.0, 2)
    >>> float(r["aic"])
    24.0
    >>> r = geron_aic([-10.0, -9.0], [2, 4])
    >>> [float(v) for v in r["aic"]]
    [24.0, 26.0]
    >>> [round(float(d), 4) for d in r["delta"]]
    [0.0, 2.0]

    References
    ----------
    Géron Ch 8
    """
    ll = np.atleast_1d(np.asarray(log_lik, dtype=float))
    kk = np.atleast_1d(np.asarray(k, dtype=float))
    if ll.size == 0:
        raise ValueError("geron_aic: log_lik is empty")
    if kk.size == 1 and ll.size > 1:
        kk = np.repeat(kk, ll.size)
    if ll.size == 1 and kk.size > 1:
        ll = np.repeat(ll, kk.size)
    if ll.shape != kk.shape:
        raise ValueError(f"geron_aic: log_lik shape {ll.shape} does not match k shape {kk.shape}")
    if not np.all(np.isfinite(ll)):
        raise ValueError("geron_aic: log_lik contains non-finite values")
    if np.any(kk < 0) or np.any(kk != np.floor(kk)):
        raise ValueError("geron_aic: k must be a non-negative integer count of free parameters")

    aic = -2.0 * ll + 2.0 * kk

    aicc = np.full_like(aic, np.nan)
    if n is not None:
        nn = float(n)
        if nn <= 0:
            raise ValueError("geron_aic: n must be a positive sample size")
        denom = nn - kk - 1.0
        # AICc is only defined while the correction denominator is positive;
        # entries where it is not stay NaN and are reported as such.
        ok = denom > 0
        aicc[ok] = aic[ok] + (2.0 * kk[ok] * (kk[ok] + 1.0)) / denom[ok]

    delta = aic - float(np.min(aic))
    w = np.exp(-0.5 * delta)
    weights = w / float(np.sum(w))
    best = int(np.argmin(aic))

    scalar = ll.size == 1
    return RichResult(
        title="Akaike information criterion",
        summary_lines=[("AIC", float(aic[best])), ("Best model index", best)],
        payload={
            "aic": float(aic[0]) if scalar else aic,
            "aicc": float(aicc[0]) if scalar else aicc,
            "delta": float(delta[0]) if scalar else delta,
            "weights": float(weights[0]) if scalar else weights,
            "best_index": best,
            "k": float(kk[0]) if scalar else kk,
            "estimate": float(aic[best]),
            "n": int(ll.size),
            "method": "Akaike information criterion (AIC = -2 log L + 2k)",
        },
    )


def cheatsheet():
    return "hmaic: Akaike information criterion for cluster-number selection"
