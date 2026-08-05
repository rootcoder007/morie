# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Krippendorff's alpha reliability.

Krippendorff (2004), *Content Analysis: An Introduction to Its
Methodology*, 2nd ed., Sage, chapter 11; the coincidence-matrix form

    alpha = 1 - D_o / D_e,
    D_o = (1/n) sum_c sum_k o_ck delta^2(c, k),
    D_e = (1/(n(n-1))) sum_c sum_k n_c n_k delta^2(c, k),

with the coincidence matrix built as o_ck = sum_u (pairs of c and k in
unit u) / (m_u - 1) over units having at least two values.  The
difference function delta^2 is what encodes the level of measurement:
nominal 0/1, interval (c - k)^2, ratio ((c - k)/(c + k))^2, ordinal the
squared difference of cumulative ranks.  Missing values are dropped
per unit, which is the whole point of the coincidence construction.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["krippendorff_alpha"]

LEVELS = ("nominal", "ordinal", "interval", "ratio")


def _delta2(level, vals, nc, i, j):
    c = vals[i]
    k = vals[j]
    if level == "nominal":
        return 0.0 if i == j else 1.0
    if level == "interval":
        return (c - k) * (c - k)
    if level == "ratio":
        if c + k == 0:
            return 0.0
        return ((c - k) / (c + k)) ** 2
    lo = min(i, j)
    hi = max(i, j)
    s = 0.0
    for g in range(lo, hi + 1):
        s += nc[g]
    s = s - (nc[i] + nc[j]) / 2.0
    return s * s


def krippendorff_alpha(data, level="nominal"):
    """Alpha for a coders-by-units reliability matrix.

    Parameters
    ----------
    data : m x N matrix
        Rows are coders, columns units.  Use nan for a missing value.
    level : one of nominal, ordinal, interval, ratio.
    """
    if level not in LEVELS:
        raise ValueError("krippendorff_alpha: level must be one of " + ", ".join(LEVELS))
    M = core.mat(data)
    if len(M) == 0 or len(M[0]) == 0:
        raise ValueError("krippendorff_alpha: data is empty")
    N = len(M[0])
    units = []
    for u in range(N):
        col = [M[i][u] for i in range(len(M)) if M[i][u] == M[i][u]]
        if len(col) >= 2:
            units.append(col)
    if not units:
        raise ValueError("krippendorff_alpha: no unit has two or more values")
    vals = sorted(set(v for col in units for v in col))
    pos = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    o = [[0.0] * V for _ in range(V)]
    for col in units:
        mu = len(col)
        for x in col:
            for y in col:
                if x is y:
                    continue
        for ai in range(mu):
            for bi in range(mu):
                if ai == bi:
                    continue
                o[pos[col[ai]]][pos[col[bi]]] += 1.0 / (mu - 1.0)
    nc = [sum(o[i]) for i in range(V)]
    n = sum(nc)
    if n <= 1:
        raise ValueError("krippendorff_alpha: fewer than two pairable values")
    do = 0.0
    de = 0.0
    for i in range(V):
        for j in range(V):
            d2 = _delta2(level, vals, nc, i, j)
            do += o[i][j] * d2
            de += nc[i] * nc[j] * d2
    do = do / n
    de = de / (n * (n - 1.0))
    a = 1.0 if de == 0.0 else 1.0 - do / de
    return RichResult(
        title="Krippendorff's alpha",
        summary_lines=[("units", len(units)), ("level", level)],
        payload={
            "estimate": a,
            "alpha": a,
            "D_o": do,
            "D_e": de,
            "n_pairable": n,
            "n_units": len(units),
            "method": "alpha = 1 - D_o/D_e on the coincidence matrix, Krippendorff (2004) ch. 11",
        },
    )


def cheatsheet():
    return "krpalp: Krippendorff's alpha"
