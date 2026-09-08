# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoeffding inequality for bounded variables."""

import math

from ._richresult import RichResult

__all__ = ["wasserman_hoeffding"]


def wasserman_hoeffding(n, t, a, b):
    """
    Hoeffding bound for the mean of i.i.d. variables in [a, b].

    Formula: P(|X_bar - mu| > t) <= 2 exp(-2 n t^2 / (b - a)^2).
    Two-sided form; the payload also carries the one-sided bound
    (without the factor 2) and the capped-at-1 version of each.

    Parameters
    ----------
    n : int
        Sample size, >= 1.
    t : float
        Deviation, > 0.
    a, b : float
        Support bounds with a < b.

    Returns
    -------
    result : dict
        Keys: estimate (two-sided, capped at 1), two_sided_raw,
        one_sided (capped), one_sided_raw, n, t, a, b, method.

    References
    ----------
    Wasserman (2004), Ch 4, Theorem 4.5 (Hoeffding).

    Examples
    --------
    n=100, t=0.1 on [0,1]: 2 exp(-2) = 0.2706705664732254.

    >>> out = wasserman_hoeffding(100, 0.1, 0.0, 1.0)
    >>> out["estimate"]
    0.2706705664732254
    >>> out["one_sided_raw"]
    0.1353352832366127
    >>> wasserman_hoeffding(1, 0.01, 0.0, 1.0)["estimate"]
    1.0
    >>> wasserman_hoeffding(10, 0.1, 1.0, 1.0)
    Traceback (most recent call last):
        ...
    ValueError: Hoeffding needs a < b; got a=1.0, b=1.0.
    """
    n = int(n)
    t = float(t)
    a = float(a)
    b = float(b)
    if n < 1:
        raise ValueError(f"Hoeffding needs n >= 1; got {n}.")
    if t <= 0:
        raise ValueError(f"Hoeffding needs t > 0; got {t}.")
    if not a < b:
        raise ValueError(f"Hoeffding needs a < b; got a={a}, b={b}.")
    expo = math.exp(-2.0 * n * t * t / (b - a) ** 2)
    return RichResult(payload={
        "estimate": float(min(2.0 * expo, 1.0)),
        "two_sided_raw": float(2.0 * expo),
        "one_sided": float(min(expo, 1.0)), "one_sided_raw": float(expo),
        "n": n, "t": t, "a": a, "b": b,
        "method": "Hoeffding 2 exp(-2 n t^2/(b-a)^2) (capped at 1)"})


def cheatsheet():
    return "wsmhfd: P(|Xbar-mu|>t) <= 2 exp(-2nt^2/(b-a)^2)"
