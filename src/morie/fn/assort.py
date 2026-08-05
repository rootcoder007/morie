# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Degree assortativity: Pearson correlation over edge endpoints.

Sources: Newman, M. E. J. (2002), "Assortative mixing in networks",
Physical Review Letters 89(20), 208701,
doi:10.1103/PhysRevLett.89.208701; and Newman, M. E. J. (2003), "Mixing
patterns in networks", Physical Review E 67(2), 026126,
doi:10.1103/PhysRevE.67.026126, fetched from arXiv (cond-mat/0209450)
and read.  Equation (26) of the 2003 paper is the computational form:

    r =  ( sum_i j_i k_i - M^-1 sum_i j_i sum_i k_i )
         -----------------------------------------------------------
         sqrt[ (sum_i j_i^2 - M^-1 (sum_i j_i)^2)
               (sum_i k_i^2 - M^-1 (sum_i k_i)^2) ]

"where j_i and k_i are the excess in-degree and out-degree of the
vertices that the ith edge leads into and out of respectively, and M is
again the number of edges. For an undirected network we can use the same
formula."

Two things decide whether this is right.

*Excess degree, or plain degree?*  The paper says excess degree, that is
degree minus one.  For an undirected graph it makes no difference: r is
a correlation, and subtracting the same constant from every j_i and
every k_i leaves it unchanged.  That invariance is checked as an anchor
rather than asserted.

*Symmetrisation.*  An undirected edge has no direction, so each edge must
enter the sums as both (j, k) and (k, j).  Skip that and r depends on how
the edge list happened to be written down -- the same graph gives
different answers.  The orientation-invariance anchor exists to catch
exactly that.

This is the scalar/degree coefficient.  It is a different quantity from
the enumerative assortativity in ``morie.fn.asorxx``; Newman derives both
and they do not agree numerically.  Neither is a duplicate of the other.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["degree_assortativity"]


def degree_assortativity(y=None, A=None, excess=True):
    """Newman's degree assortativity coefficient.

    Parameters
    ----------
    y : ignored
        Accepted for interface compatibility with the rest of the shelf.
    A : array-like
        n-by-n adjacency matrix, treated as undirected and unweighted;
        the diagonal is ignored.
    excess : bool
        Use excess degree (degree - 1), as the paper does.  The result is
        identical either way; the switch exists so the invariance can be
        exercised.

    Returns
    -------
    r : the degree assortativity coefficient
    M : the number of undirected edges
    degree : the degree sequence
    """
    if A is None:
        raise ValueError("degree_assortativity: an adjacency matrix is required")
    Am = [[float(v) for v in row] for row in A]
    n = len(Am)
    if n == 0:
        raise ValueError("degree_assortativity: the graph is empty")
    for row in Am:
        if len(row) != n:
            raise ValueError("degree_assortativity: the adjacency matrix is not square")
    deg = [0.0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if Am[i][j] != 0.0 or Am[j][i] != 0.0:
                edges.append((i, j))
                deg[i] += 1.0
                deg[j] += 1.0
    M = len(edges)
    if M == 0:
        raise ValueError("degree_assortativity: the graph has no edges")
    off = 1.0 if excess else 0.0
    sjk = 0.0
    sj = 0.0
    sk = 0.0
    sj2 = 0.0
    sk2 = 0.0
    # each undirected edge enters twice, once in each orientation
    for (u, v) in edges:
        for (p, q) in ((u, v), (v, u)):
            jj = deg[p] - off
            kk = deg[q] - off
            sjk += jj * kk
            sj += jj
            sk += kk
            sj2 += jj * jj
            sk2 += kk * kk
    m2 = 2.0 * M
    num = sjk - sj * sk / m2
    d1 = sj2 - sj * sj / m2
    d2 = sk2 - sk * sk / m2
    den = math.sqrt(d1 * d2)
    if den == 0.0:
        raise ValueError("degree_assortativity: every edge endpoint has the same degree, r is undefined")
    r = num / den
    return RichResult(
        title="Degree assortativity",
        summary_lines=[("r", r), ("M", M)],
        payload={
            "r": r,
            "estimate": r,
            "M": M,
            "degree": deg,
            "sum_jk": sjk,
            "sum_j": sj,
            "sum_j2": sj2,
            "excess": bool(excess),
            "n": n,
            "method": "Newman (2003) eq. (26), degree assortativity over symmetrised edge endpoints",
        },
    )


def cheatsheet():
    return "assort: Degree assortativity coefficient"


# compact alias per ledger/NAMING.md
degreeassortativity = degree_assortativity
