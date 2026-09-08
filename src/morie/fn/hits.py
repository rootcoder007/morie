# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""HITS: hubs and authorities.

Kleinberg (1999), "Authoritative sources in a hyperlinked
environment", J. ACM 46(5):604-632, doi:10.1145/324133.324140.
Section 3 defines the two operations

    I:  a_p <- sum_{q -> p} h_q        O:  h_p <- sum_{p -> q} a_q

applied alternately and followed by a normalisation.  In matrix form
a = A' h and h = A a, so a converges to the principal eigenvector of
A'A and h to the principal eigenvector of A A'.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hits"]


def _unit(v):
    s = math.sqrt(sum(x * x for x in v))
    if s == 0.0:
        return list(v)
    return [x / s for x in v]


def _square(A, who):
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError(who + ": adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError(who + ": adjacency matrix must be square")
    return M, n


def hits(A, iters=50):
    """Hub and authority vectors of a directed adjacency matrix.

    Parameters
    ----------
    A : n x n array-like
        A[i][j] > 0 when there is a link i -> j.
    iters : int
        Number of I/O sweeps.  Must be at least one.

    Returns
    -------
    estimate : the largest hub score
    hubs, authorities : the two unit-L2 score vectors
    """
    M, n = _square(A, "hits")
    m = int(iters)
    if m < 1:
        raise ValueError("hits: iters must be at least 1")
    h = _unit([1.0] * n)
    a = [0.0] * n
    for _ in range(m):
        a = _unit([sum(M[i][j] * h[i] for i in range(n)) for j in range(n)])
        h = _unit([sum(M[i][j] * a[j] for j in range(n)) for i in range(n)])
    top = 0
    for i in range(n):
        if h[i] > h[top]:
            top = i
    return RichResult(
        title="HITS hubs and authorities",
        summary_lines=[("n", n), ("iters", m)],
        payload={
            "estimate": h[top],
            "hubs": h,
            "authorities": a,
            "top_hub": top + 1,
            "n": n,
            "iters": m,
            "method": "alternating I/O operations of Kleinberg (1999) sect. 3, L2-normalised",
        },
    )


def cheatsheet():
    return "hits: HITS (hubs & authorities)"
