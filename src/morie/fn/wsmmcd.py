# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""McDiarmid's inequality."""

import math

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_mcdiarmid"]


def wasserman_mcdiarmid(t, c):
    """
    McDiarmid bounded-differences inequality.

    Formula: P(|f(X) - E f(X)| > t) <= 2 exp(-2 t^2 / sum_i c_i^2),
    where c_i bounds the change in f when the i-th coordinate alone
    changes. One-sided and capped versions ship alongside. With
    f = the mean of variables in [a, b] (c_i = (b-a)/n) this
    recovers Hoeffding exactly.

    Parameters
    ----------
    t : float
        Deviation, > 0.
    c : array-like
        Bounded-difference constants, each > 0.

    Returns
    -------
    result : dict
        Keys: estimate (two-sided, capped at 1), two_sided_raw,
        one_sided, sum_c_sq, n, t, method.

    References
    ----------
    Wasserman (2004), Ch 4; McDiarmid (1989).

    Examples
    --------
    Matches Hoeffding for the mean of 100 variables in [0,1]:

    >>> out = wasserman_mcdiarmid(0.1, [0.01] * 100)
    >>> round(out["estimate"], 12)
    0.270670566473
    >>> out["sum_c_sq"]
    0.01
    >>> wasserman_mcdiarmid(0.1, [0.01, -0.02])
    Traceback (most recent call last):
        ...
    ValueError: bounded-difference constants must be positive.
    """
    t = float(t)
    c = np.atleast_1d(np.asarray(c, dtype=float))
    if t <= 0:
        raise ValueError(f"McDiarmid needs t > 0; got {t}.")
    if np.any(c <= 0):
        raise ValueError("bounded-difference constants must be positive.")
    s = float(np.sum(c ** 2))
    expo = math.exp(-2.0 * t * t / s)
    return RichResult(payload={
        "estimate": float(min(2.0 * expo, 1.0)),
        "two_sided_raw": float(2.0 * expo),
        "one_sided": float(min(expo, 1.0)), "sum_c_sq": s,
        "n": int(c.size), "t": t,
        "method": "McDiarmid 2 exp(-2t^2 / sum c_i^2) (capped at 1)"})


def cheatsheet():
    return "wsmmcd: bound 2 exp(-2t^2/sum c^2); c_i = (b-a)/n recovers Hoeffding"
