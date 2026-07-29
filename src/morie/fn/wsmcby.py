# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chebyshev inequality P(|X-mu|>=k sigma) <= 1/k^2."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wasserman_chebyshev_ineq"]


def wasserman_chebyshev_ineq(k):
    """
    Chebyshev inequality P(|X - mu| >= k sigma) <= 1/k^2.

    Formula: bound = 1/k^2 for each k > 0. The bound is trivial
    (capped at 1) for k <= 1; k <= 0 is refused because the event
    "at least zero standard deviations away" has probability 1 and
    the inequality carries no content there.

    Parameters
    ----------
    k : array-like
        Number(s) of standard deviations, each strictly positive.

    Returns
    -------
    result : dict
        Keys: estimate (bound for the first k), bounds (per k,
        capped at 1), raw_bounds (1/k^2 uncapped), k, n, method.

    References
    ----------
    Wasserman (2004), Ch 4, Theorem 4.2.

    Examples
    --------
    >>> out = wasserman_chebyshev_ineq(2.0)
    >>> out["estimate"]
    0.25
    >>> wasserman_chebyshev_ineq(3.0)["estimate"]
    0.1111111111111111
    >>> wasserman_chebyshev_ineq(0.5)["bounds"]
    [1.0]
    >>> wasserman_chebyshev_ineq(0.5)["raw_bounds"]
    [4.0]
    >>> wasserman_chebyshev_ineq(0.0)
    Traceback (most recent call last):
        ...
    ValueError: Chebyshev inequality needs k > 0; got 0.0.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if np.any(k <= 0):
        bad = float(k[k <= 0][0])
        raise ValueError(f"Chebyshev inequality needs k > 0; got {bad}.")
    raw = 1.0 / k ** 2
    capped = np.minimum(raw, 1.0)
    return RichResult(payload={
        "estimate": float(capped[0]),
        "bounds": [float(v) for v in capped],
        "raw_bounds": [float(v) for v in raw],
        "k": [float(v) for v in k], "n": int(k.size),
        "method": "Chebyshev bound 1/k^2 (capped at 1)"})


def cheatsheet():
    return "wsmcby: P(|X-mu| >= k sigma) <= 1/k^2, k > 0"
