# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Empirical quantile q_p = inf{x : F_n(x) >= p}."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_empirical_quantile"]


def wasserman_empirical_quantile(data, p):
    """
    Empirical quantile via the left-continuous inverse of the eCDF.

    Formula: q_p = inf{x : F_n(x) >= p}. This is numpy's
    interpolation='lower'-free DEFINITION-faithful version: the
    returned value is always an order statistic X_(ceil(np)), never
    an interpolation, matching Wasserman's inf definition (and R's
    type-1 quantile).

    Parameters
    ----------
    data : array-like
        Sample (non-empty).
    p : array-like
        Probability level(s) in (0, 1].

    Returns
    -------
    result : dict
        Keys: estimate (q at the first p), values (per p), p, n,
        method.

    References
    ----------
    Wasserman (2004), Ch 7 (sample quantiles); Hyndman-Fan type 1.

    Examples
    --------
    >>> d = [3.0, 1.0, 4.0, 2.0]
    >>> wasserman_empirical_quantile(d, 0.5)["estimate"]
    2.0
    >>> wasserman_empirical_quantile(d, 0.51)["estimate"]
    3.0
    >>> wasserman_empirical_quantile(d, [0.25, 1.0])["values"]
    [1.0, 4.0]
    >>> wasserman_empirical_quantile(d, 0.0)
    Traceback (most recent call last):
        ...
    ValueError: quantile levels must lie in (0, 1]; got 0.0.
    """
    data = np.sort(np.atleast_1d(np.asarray(data, dtype=float)))
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = data.size
    if n == 0:
        raise ValueError("the quantile of an empty sample is undefined.")
    if np.any((p <= 0) | (p > 1)):
        bad = float(p[(p <= 0) | (p > 1)][0])
        raise ValueError(f"quantile levels must lie in (0, 1]; got {bad}.")
    idx = np.ceil(p * n).astype(int) - 1
    vals = data[idx]
    return RichResult(payload={
        "estimate": float(vals[0]), "values": [float(v) for v in vals],
        "p": [float(v) for v in p], "n": int(n),
        "method": "type-1 quantile q_p = X_(ceil(np))"})


def cheatsheet():
    return "wsmqtl: q_p = inf{x: F_n(x) >= p} = X_(ceil(np)), type-1, no interpolation"
