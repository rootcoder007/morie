# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chi-square goodness-of-fit test."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_chi_sq_gof"]


def _chi2_sf(x, k):
    """P(Chi2_k > x) via the regularised upper incomplete gamma,
    series/continued-fraction split (stdlib only, ~1e-12 accurate)."""
    a = k / 2.0
    x = x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        # lower series
        term = 1.0 / a
        total = term
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        p_lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p_lower
    # upper continued fraction (Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def wasserman_chi_sq_gof(observed, expected):
    """
    Chi-square goodness-of-fit statistic and p-value.

    Formula: X^2 = sum_j (O_j - E_j)^2 / E_j, compared to Chi^2 with
    k - 1 degrees of freedom (no estimated parameters). The p-value
    uses a stdlib incomplete-gamma survival function. Expected
    counts must be positive; a mismatch between total observed and
    total expected is reported (not silently rescaled).

    Parameters
    ----------
    observed : array-like
        Observed counts (>= 0), length k >= 2.
    expected : array-like
        Expected counts (> 0), same length.

    Returns
    -------
    result : dict
        Keys: estimate (X^2), p_value, df, per_cell, total_observed,
        total_expected, k, method.

    References
    ----------
    Wasserman (2004), Ch 10, section 10.8.

    Examples
    --------
    >>> out = wasserman_chi_sq_gof([10, 20, 30], [20, 20, 20])
    >>> out["estimate"]
    10.0
    >>> out["df"]
    2
    >>> abs(out["p_value"] - 0.006737946999085467) < 1e-12
    True
    >>> wasserman_chi_sq_gof([5, 5], [5, 0])
    Traceback (most recent call last):
        ...
    ValueError: expected counts must be strictly positive.
    """
    obs = np.atleast_1d(np.asarray(observed, dtype=float))
    exp_ = np.atleast_1d(np.asarray(expected, dtype=float))
    if obs.size != exp_.size:
        raise ValueError(f"observed ({obs.size}) and expected ({exp_.size}) lengths differ.")
    k = obs.size
    if k < 2:
        raise ValueError("a goodness-of-fit test needs at least 2 cells.")
    if np.any(obs < 0):
        raise ValueError("observed counts cannot be negative.")
    if np.any(exp_ <= 0):
        raise ValueError("expected counts must be strictly positive.")
    per = (obs - exp_) ** 2 / exp_
    stat = float(np.sum(per))
    df = k - 1
    return RichResult(payload={
        "estimate": stat, "p_value": float(_chi2_sf(stat, df)),
        "df": int(df), "per_cell": [float(v) for v in per],
        "total_observed": float(np.sum(obs)),
        "total_expected": float(np.sum(exp_)), "k": int(k),
        "method": "chi-square GOF sum (O-E)^2/E vs Chi2_{k-1}"})


def cheatsheet():
    return "wsmchi: X^2 = sum (O-E)^2/E, df = k-1, stdlib igamma p-value"
