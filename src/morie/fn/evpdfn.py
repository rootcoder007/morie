# morie.fn -- function file (rootcoder007/morie)
"""Empirical Pickands dependence function."""

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["evt_pickands_dep_fn"]


def evt_pickands_dep_fn(x, y, t_grid=None, u=None):
    """
    Empirical Pickands dependence function

    Formula: A(t) = -log(P(F_X(X) <= u^(1-t), F_Y(Y) <= u^t)) / (-log u)

    A(t) is read off the empirical copula at the diagonal-shifted point
    (u^(1-t), u^t).  It satisfies A(0) = A(1) = 1 exactly by
    construction, lies between max(t, 1-t) and 1, equals 1 under
    independence and max(t, 1-t) under perfect dependence.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    t_grid : array-like or None
        Points in [0, 1].  None uses eleven equally spaced points.
    u : float or None
        Copula level in (0, 1).  None uses exp(-1).

    Returns
    -------
    result : dict
        Keys: A, t, estimate (A at 1/2), chi, convex_ok, n.

    References
    ----------
    Pickands (1981), Bull. Int. Statist. Inst. 49:859-878.
    """
    xs = core.vec(x)
    ys = core.vec(y)
    n = len(xs)
    if n == 0:
        raise ValueError("empty input: x has no observations")
    if len(ys) != n:
        raise ValueError("x and y must have the same length")
    if t_grid is None:
        t_grid = [i / 10.0 for i in range(11)]
    else:
        t_grid = core.vec(t_grid)
    if any(v < 0.0 or v > 1.0 for v in t_grid):
        raise ValueError("t_grid must lie in [0, 1]")
    if u is None:
        u = math.exp(-1.0)
    u = float(u)
    if not (0.0 < u < 1.0):
        raise ValueError("u must lie strictly in (0, 1)")
    # marginal ranks -> pseudo-observations, with the n+1 divisor
    rx = core.rank_avg(xs)
    ry = core.rank_avg(ys)
    ux = [v / (n + 1.0) for v in rx]
    uy = [v / (n + 1.0) for v in ry]
    A, lu = [], -math.log(u)
    for t in t_grid:
        a = u ** (1.0 - t)
        b = u ** t
        c = 0
        for i in range(n):
            if ux[i] <= a and uy[i] <= b:
                c += 1
        p = c / float(n)
        if t <= 0.0:
            A.append(1.0)
        elif t >= 1.0:
            A.append(1.0)
        elif p <= 0.0:
            A.append(1.0)
        else:
            v = -math.log(p) / lu
            lower = max(t, 1.0 - t)
            A.append(min(max(v, lower), 1.0))
    half = None
    for i, t in enumerate(t_grid):
        if abs(t - 0.5) < 1e-12:
            half = A[i]
    if half is None:
        half = sum(A) / len(A)
    convex = 1
    for i in range(1, len(A) - 1):
        if A[i] > 0.5 * (A[i - 1] + A[i + 1]) + 1e-9:
            convex = 0
    return RichResult(payload={
        "A": A,
        "t": list(t_grid),
        "estimate": half,
        "chi": 2.0 - 2.0 * half,
        "convex_ok": convex,
        "n": n,
        "method": "empirical Pickands dependence function",
    })


def cheatsheet():
    return "evpdfn: empirical Pickands dependence function"


# compact alias per ledger/NAMING.md
evtpickandsdepfn = evt_pickands_dep_fn
