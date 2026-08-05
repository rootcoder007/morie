# morie.fn -- wave 2 slice f_00 (rootcoder007/morie)
"""Assortativity coefficient for an enumerative (categorical) attribute.

Source: Newman, M. E. J. (2003), "Mixing patterns in networks",
Physical Review E 67(2), 026126, doi:10.1103/PhysRevE.67.026126, fetched
from arXiv (cond-mat/0209450) and read.  Equation (2) on p.2:

    r = ( sum_i e_ii - sum_i a_i b_i ) / ( 1 - sum_i a_i b_i )

where e_ij is the fraction of all edges joining a vertex of type i to a
vertex of type j, a_i = sum_j e_ij and b_i = sum_j e_ji.  The paper also
gives the lower bound, equation (3),

    r_min = - sum_i a_i b_i / ( 1 - sum_i a_i b_i ),

which is returned as ``r_min``: r is *not* a correlation on [-1, 1].  It
reaches 1 for perfect assortative mixing but its most negative
attainable value depends on the type distribution, so calling a value of
-0.3 "weak disassortativity" without comparing it to r_min is a mistake
the output makes it possible to avoid.  ``r_normalised`` = r / |r_min|
is supplied for that comparison when r < 0.

This is the **enumerative** coefficient, for unordered categories.  It is
a different quantity from the degree/scalar assortativity in
``morie.fn.assort``, which is a Pearson correlation over edge endpoints;
Newman derives both in this one paper and they do not agree numerically.
Neither is a duplicate of the other.

The mixing matrix is symmetrised: an undirected edge between types i and
j contributes half to e_ij and half to e_ji, so e is symmetric and sums
to one regardless of how the edge list happened to be oriented.  Without
that, r depends on arbitrary orientation.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core  # noqa: F401

from ._richresult import RichResult

__all__ = ["assortativity"]


def assortativity(G, attribute):
    """Newman's enumerative assortativity coefficient.

    Parameters
    ----------
    G : array-like
        n-by-n adjacency matrix.  Treated as undirected: entry (i, j)
        and (j, i) both count, and the diagonal (self-loops) is ignored.
        Non-zero entries are edges; weights are honoured.
    attribute : array-like
        Categorical vertex label, one per vertex.

    Returns
    -------
    r : the assortativity coefficient
    r_min : the most negative value attainable for this type distribution
    e : the mixing matrix
    a, b : its row and column sums
    """
    A = [[float(v) for v in row] for row in G]
    n = len(A)
    if n == 0:
        raise ValueError("assortativity: the graph is empty")
    for row in A:
        if len(row) != n:
            raise ValueError("assortativity: the adjacency matrix is not square")
    att = list(attribute)
    if len(att) != n:
        raise ValueError("assortativity: attribute has one entry per vertex")
    types = []
    for v in att:
        if str(v) not in [str(t) for t in types]:
            types.append(v)
    types = sorted(types, key=lambda v: str(v))
    T = len(types)
    if T == 0:
        raise ValueError("assortativity: no attribute values")
    pos = {}
    for i, t in enumerate(types):
        pos[str(t)] = i
    e = [[0.0] * T for _ in range(T)]
    tot = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = A[i][j]
            if w == 0.0:
                continue
            if w < 0.0:
                raise ValueError("assortativity: negative edge weight")
            ti = pos[str(att[i])]
            tj = pos[str(att[j])]
            # symmetrise: each ordered pair contributes half to each direction
            e[ti][tj] += 0.5 * w
            e[tj][ti] += 0.5 * w
            tot += w
    if tot <= 0.0:
        raise ValueError("assortativity: the graph has no edges")
    for i in range(T):
        for j in range(T):
            e[i][j] = e[i][j] / tot
    a = []
    b = []
    for i in range(T):
        sa = 0.0
        sb = 0.0
        for j in range(T):
            sa += e[i][j]
            sb += e[j][i]
        a.append(sa)
        b.append(sb)
    tr = 0.0
    for i in range(T):
        tr += e[i][i]
    ab = 0.0
    for i in range(T):
        ab += a[i] * b[i]
    den = 1.0 - ab
    if den == 0.0:
        raise ValueError("assortativity: every edge is within one type, r is undefined")
    r = (tr - ab) / den
    rmin = -ab / den
    return RichResult(
        title="Enumerative assortativity",
        summary_lines=[("r", r), ("types", T)],
        payload={
            "r": r,
            "estimate": r,
            "r_min": rmin,
            "r_normalised": (r / abs(rmin)) if (r < 0.0 and rmin != 0.0) else float("nan"),
            "e": e,
            "a": a,
            "b": b,
            "trace_e": tr,
            "sum_ab": ab,
            "n_types": T,
            "n": n,
            "method": "Newman (2003) eq. (2), enumerative assortativity",
        },
    )


def cheatsheet():
    return "asorxx: Enumerative assortativity coefficient"
