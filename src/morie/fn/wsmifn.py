# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empirical influence (sensitivity curve)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_influence_function"]


def wasserman_influence_function(data, T):
    """
    Empirical influence values via the sensitivity curve.

    Formula: L_F(x) = lim_{e->0} (T((1-e)F + e delta_x) - T(F)) / e.
    With F = F_n and the smallest natural contamination e = 1/(n+1)
    (append one copy of x to the sample), the finite-difference
    version is the sensitivity curve
    SC(x) = (n+1) (T(x_1..x_n, x) - T(x_1..x_n)),
    evaluated here at each observed x_i. For T = mean this equals
    x_i - x_bar exactly, the textbook influence function of the mean.

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    T : callable or None
        Statistic (None = sample mean).

    Returns
    -------
    result : dict
        Keys: estimate (T on the data), influence (per observation),
        epsilon, mean_influence, n, method.

    References
    ----------
    Wasserman (2004), Ch 7 (statistical functionals); Hampel's
    sensitivity curve.

    Examples
    --------
    >>> out = wasserman_influence_function([1.0, 2.0, 3.0, 4.0], None)
    >>> out["estimate"]
    2.5
    >>> [round(v, 12) for v in out["influence"]]
    [-1.5, -0.5, 0.5, 1.5]
    >>> round(out["mean_influence"], 12)
    0.0
    >>> wasserman_influence_function([], None)
    Traceback (most recent call last):
        ...
    ValueError: the influence function of an empty sample is undefined.
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    if n == 0:
        raise ValueError("the influence function of an empty sample is undefined.")
    if T is None:
        T = lambda a: float(np.mean(a))
    base = float(T(data))
    infl = [float((n + 1) * (float(T(np.append(data, xi))) - base))
            for xi in data]
    return RichResult(payload={
        "estimate": base, "influence": infl,
        "epsilon": 1.0 / (n + 1),
        "mean_influence": float(np.mean(infl)), "n": int(n),
        "method": "sensitivity curve SC(x) = (n+1)(T(data+x) - T(data))"})


def cheatsheet():
    return "wsmifn: SC(x_i) = (n+1)(T(data ∪ x_i) - T(data)); mean -> x_i - xbar"
