# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empirical distribution function (eCDF)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_empirical_cdf"]


def wasserman_empirical_cdf(x, data):
    """
    Empirical distribution function F_n at the point(s) x.

    Formula: F_n(x) = (1/n) sum_i I(X_i <= x), right-continuous with
    the <= convention, so F_n evaluated AT an observation includes it.

    Parameters
    ----------
    x : array-like
        Evaluation point(s).
    data : array-like
        The observed sample X_1..X_n (non-empty).

    Returns
    -------
    result : dict
        Keys: estimate (F_n at the first x), values (per x), x, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 7, Definition 7.1.

    Examples
    --------
    >>> d = [1.0, 2.0, 3.0, 4.0]
    >>> wasserman_empirical_cdf(2.5, d)["estimate"]
    0.5
    >>> wasserman_empirical_cdf(2.0, d)["estimate"]
    0.5
    >>> wasserman_empirical_cdf([0.0, 4.0, 9.9], d)["values"]
    [0.0, 1.0, 1.0]
    >>> wasserman_empirical_cdf(1.0, [])
    Traceback (most recent call last):
        ...
    ValueError: the eCDF of an empty sample is undefined.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = data.size
    if n == 0:
        raise ValueError("the eCDF of an empty sample is undefined.")
    vals = np.searchsorted(np.sort(data), x, side="right") / float(n)
    return RichResult(payload={
        "estimate": float(vals[0]), "values": [float(v) for v in vals],
        "x": [float(v) for v in x], "n": int(n),
        "method": "eCDF F_n(x) = (1/n) sum I(X_i <= x)"})


def cheatsheet():
    return "wsmcdf: F_n(x) = (1/n) sum I(X_i <= x), right-continuous"
