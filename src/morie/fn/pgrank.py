# morie.fn -- function file (rootcoder007/morie)
"""PageRank with a fixed power iteration."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["pagerank"]


def pagerank(A, d=0.85, n_iter=100):
    """Stationary importance of a node under a random surfer.

    Two things make the recursion well posed rather than circular.  The
    damping factor gives every node a floor, so a dangling region cannot
    absorb all the mass; and dangling nodes -- pages with no out-links --
    have their mass redistributed uniformly instead of leaking, which is
    the detail that decides whether the vector sums to one.

    Determinism: a fixed number of power iterations, no tolerance test.

    Formula: ``PR(v) = (1 - d) / n + d sum_{u -> v} PR(u) / L(u)``,
    with the mass of dangling ``u`` spread over all nodes.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Adjacency matrix; ``A[i][j]`` non-zero means a link from i to j.
    d : float, default 0.85
        Damping factor.
    n_iter : int, default 100
        Power iterations.

    Returns
    -------
    RichResult
        ``pr``, ``estimate`` (the largest score), ``top`` (its index),
        ``n``.

    References
    ----------
    Page, L., Brin, S., Motwani, R. & Winograd, T. (1999).  The PageRank
    citation ranking: bringing order to the web.  Stanford InfoLab
    technical report 1999-66.
    """
    M = C.mat(A)
    n = len(M)
    out = [sum(M[i]) for i in range(n)]
    pr = [1.0 / n] * n
    for _ in range(int(n_iter)):
        dangle = sum(pr[i] for i in range(n) if out[i] == 0.0) / n
        new = []
        for j in range(n):
            s = sum(pr[i] * M[i][j] / out[i] for i in range(n) if out[i] > 0.0)
            new.append((1.0 - d) / n + d * (s + dangle))
        pr = new
    top = max(range(n), key=lambda i: pr[i])
    return RichResult(payload={
        "pr": pr, "estimate": pr[top], "top": top, "n": n,
        "method": "PageRank by fixed power iteration"})


def cheatsheet():
    return "pgrank: PageRank with a fixed power iteration."
