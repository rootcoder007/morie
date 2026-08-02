# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Moment generating function M_X(t) = E[e^{tX}]."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_mgf"]


def wasserman_mgf(x, t):
    """
    Empirical moment generating function.

    Formula: M_X(t) = E[e^{tX}], estimated by (1/n) sum_i e^{t X_i}
    at each requested t. M(0) = 1 exactly, a built-in sanity anchor.

    Parameters
    ----------
    x : array-like
        Sample (non-empty).
    t : array-like
        Evaluation point(s).

    Returns
    -------
    result : dict
        Keys: estimate (M at the first t), values (per t), t, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 3.

    Examples
    --------
    >>> wasserman_mgf([1.0, 2.0], 0.0)["estimate"]
    1.0
    >>> import math
    >>> out = wasserman_mgf([0.0, 1.0], 1.0)
    >>> abs(out["estimate"] - (1.0 + math.e) / 2.0) < 1e-15
    True
    >>> wasserman_mgf([], 1.0)
    Traceback (most recent call last):
        ...
    ValueError: the MGF of an empty sample is undefined.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if x.size == 0:
        raise ValueError("the MGF of an empty sample is undefined.")
    vals = [float(np.mean(np.exp(ti * x))) for ti in t]
    return RichResult(payload={
        "estimate": vals[0], "values": vals,
        "t": [float(v) for v in t], "n": int(x.size),
        "method": "empirical MGF (1/n) sum e^{tX_i}"})


def cheatsheet():
    return "wsmmgf: M(t) = (1/n) sum e^{t X_i}; M(0) = 1"
