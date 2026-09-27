# morie.fn -- shared core for transferable-utility games (rootcoder007/morie)
"""Coalitions, excesses and the sequential LPs behind the (pre)nucleolus.

A game on n players is the vector ``v`` of length ``2**n - 1`` in binary
order: ``v[S - 1]`` is the worth of the coalition whose bitmask is ``S``
(player ``i`` belongs to ``S`` when bit ``i`` is set).
"""

from __future__ import annotations

import math

from ._qpcore import simplex_standard, ssum


def players(v):
    n = int(round(math.log2(len(v) + 1)))
    if 2**n - 1 != len(v):
        raise ValueError("v must have length 2**n - 1 (binary order)")
    return n


def members(S, n):
    return [i for i in range(n) if S >> i & 1]


def worth(v, S):
    return 0.0 if S == 0 else float(v[S - 1])


def excess(v, S, x):
    return worth(v, S) - ssum(x[i] for i in members(S, len(x)))


def max_surplus(v, x, i, j):
    """s_ij(x): the largest excess of a coalition containing i but not j."""
    n = len(x)
    return max(excess(v, S, x) for S in range(1, 2**n) if S >> i & 1 and not S >> j & 1)


def _rank(rows, n):
    M = [list(r) for r in rows]
    rank = 0
    for c in range(n):
        p = max(range(rank, len(M)), key=lambda r: abs(M[r][c]), default=None)
        if p is None or abs(M[p][c]) < 1e-9:
            continue
        M[rank], M[p] = M[p], M[rank]
        for r in range(len(M)):
            if r != rank and M[r][c] != 0.0:
                f = M[r][c] / M[rank][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[rank])]
        rank += 1
    return rank


def _lp(n, le, eq, cx, ct, lower):
    """min cx.x + ct*t s.t. rows (a, at, rhs): a.x + at*t <= rhs (le) or = rhs (eq).

    x_i = lower_i + p_i when ``lower`` is given, else p_i - q_i; t = t+ - t-.
    """
    free = lower is None
    nx = 2 * n if free else n
    ncol = nx + 2 + len(le)
    A, b = [], []
    for k, (a, at, rhs) in enumerate(le + eq):
        row = [0.0] * ncol
        for i in range(n):
            row[i] = a[i]
            if free:
                row[n + i] = -a[i]
        row[nx], row[nx + 1] = at, -at
        if k < len(le):
            row[nx + 2 + k] = 1.0
        A.append(row)
        b.append(rhs - (0.0 if free else ssum(a[i] * lower[i] for i in range(n))))
    c = [0.0] * ncol
    for i in range(n):
        c[i] = cx[i]
        if free:
            c[n + i] = -cx[i]
    c[nx], c[nx + 1] = ct, -ct
    z, status = simplex_standard(c, A, b)
    if status != "optimal":
        raise ValueError(f"linear program {status}")
    x = [(z[i] - z[n + i]) if free else lower[i] + z[i] for i in range(n)]
    return x, z[nx] - z[nx + 1]


def sequential_nucleolus(v, pre, tol=1e-9):
    """Maschler, Peleg and Shapley (1979) sequence of LPs; returns (x, levels)."""
    n = players(v)
    full = 2**n - 1
    ind = [[1.0 if S >> i & 1 else 0.0 for i in range(n)] for S in range(full + 1)]
    lower = None if pre else [worth(v, 1 << i) for i in range(n)]
    active = list(range(1, full))
    fixed = {}
    levels = []
    x = None
    while active:
        le = [([-a for a in ind[S]], -1.0, -worth(v, S)) for S in active]
        eq = [(ind[full], 0.0, worth(v, full))] + [
            ([-a for a in ind[S]], 0.0, e - worth(v, S)) for S, e in fixed.items()
        ]
        x, t = _lp(n, le, eq, [0.0] * n, 1.0, lower)
        levels.append(t)
        cap = [([-a for a in ind[S]], 0.0, t - worth(v, S)) for S in active]
        newly = []
        for S in active:
            if excess(v, S, x) < t - tol:
                continue
            y, _ = _lp(n, cap, eq, [-a for a in ind[S]], 0.0, lower)
            if excess(v, S, y) >= t - tol:
                newly.append(S)
        if not newly:
            raise ValueError("no coalition could be fixed; tolerance too tight")
        for S in newly:
            fixed[S] = t
        span = [ind[full]] + [ind[S] for S in fixed]
        r = _rank(span, n)
        if r == n:
            break
        active = [S for S in active if S not in fixed and _rank(span + [ind[S]], n) > r]
    return x, levels
