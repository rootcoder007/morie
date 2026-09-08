# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""HITS iterated to convergence rather than for a fixed sweep count.

Kleinberg (1999), J. ACM 46(5):604-632, doi:10.1145/324133.324140.
Same I and O operations as :mod:`hits`; here the loop stops when the
largest absolute change in the normalised hub vector falls below tol,
which is the "converge" wording of the paper's Iterate procedure.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["hits_hubs_authorities"]


def _unit(v):
    s = math.sqrt(sum(x * x for x in v))
    if s == 0.0:
        return list(v)
    return [x / s for x in v]


def hits_hubs_authorities(y, A, tol=1e-12, max_iter=1000):
    """Run the HITS recursion from start vector y until it settles.

    Parameters
    ----------
    y : length-n array-like
        Starting hub vector; the paper starts from all ones.
    A : n x n array-like
        Adjacency matrix, A[i][j] > 0 for a link i -> j.
    tol : float
        Stop when max_i |h_i^(t) - h_i^(t-1)| <= tol.
    """
    M = core.mat(A)
    n = len(M)
    if n == 0:
        raise ValueError("hits_hubs_authorities: adjacency matrix is empty")
    for r in M:
        if len(r) != n:
            raise ValueError("hits_hubs_authorities: adjacency matrix must be square")
    h = core.vec(y)
    if len(h) != n:
        raise ValueError("hits_hubs_authorities: y and A have different lengths")
    if tol <= 0:
        raise ValueError("hits_hubs_authorities: tol must be positive")
    h = _unit(h)
    a = [0.0] * n
    it = 0
    delta = float("inf")
    while it < int(max_iter) and delta > tol:
        a = _unit([sum(M[i][j] * h[i] for i in range(n)) for j in range(n)])
        hn = _unit([sum(M[i][j] * a[j] for j in range(n)) for i in range(n)])
        delta = max(abs(hn[i] - h[i]) for i in range(n))
        h = hn
        it += 1
    return RichResult(
        title="HITS run to convergence",
        summary_lines=[("n", n), ("iterations", it)],
        payload={
            "estimate": max(h),
            "hubs": h,
            "authorities": a,
            "iterations": it,
            "delta": delta,
            "converged": bool(delta <= tol),
            "n": n,
            "method": "Kleinberg (1999) I/O recursion, stopped at max|dh| <= tol",
        },
    )
