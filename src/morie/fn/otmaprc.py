# morie.fn -- function file (rootcoder007/morie)
"""Brenier optimal transport map in one dimension."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['brenier1d', 'ot_map_recovery_brenier']


def brenier1d(x, y, p=2.0):
    """Brenier optimal transport map in one dimension.

    In one dimension Brenier's theorem has a closed form -- the optimal map for any strictly convex cost is the monotone one, so sorting both samples and pairing them rank by rank solves the problem exactly, with no iteration and no regularisation. That makes it the right thing to test a general solver against.


    Formula: T(x_(i)) = y_(i): the monotone rearrangement matching order statistic to order statistic

    Parameters
    ----------
    x : array-like
        Source sample.
    y : array-like
        Target sample, the same length.
    p : float
        Cost exponent |x - y|^p.

    Returns
    -------
    RichResult
        ``map`` (image of each input in its original order), ``cost``, ``order_x``, ``order_y``, ``n``.

    References
    ----------
    Brenier (1991), Polar factorization and monotone rearrangement of
    vector-valued functions, Communications on Pure and Applied
    Mathematics 44:375-417.  Not held locally; the one-dimensional
    monotone-rearrangement solution is the standard published result.
    """
    x = C.vec(x); y = C.vec(y)
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    ox = sorted(range(n), key=lambda i: (x[i], i))
    oy = sorted(range(n), key=lambda i: (y[i], i))
    mp = [0.0] * n
    for k in range(n):
        mp[ox[k]] = y[oy[k]]
    cost = sum(abs(x[i] - mp[i]) ** float(p) for i in range(n)) / n
    return RichResult(payload={
        "map": mp, "cost": cost, "order_x": ox, "order_y": oy, "n": n,
        "method": "Brenier map in one dimension (monotone rearrangement)"})


ot_map_recovery_brenier = brenier1d


def cheatsheet():
    return "otmaprc: Brenier optimal transport map in one dimension."
