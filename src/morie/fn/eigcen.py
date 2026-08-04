# morie.fn -- function file (rootcoder007/morie)
"""Eigenvector centrality."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['eigcent', 'eigenvector_centrality']


def eigcent(A):
    """Eigenvector centrality.

    A vertex is important when its neighbours are important, which is a fixed point rather than a count, and the fixed point is the leading eigenvector. Two scalings are returned: unit Euclidean norm and maximum entry one. The eigenvector is sign-fixed on its largest-magnitude entry and then, since a centrality cannot be negative, flipped so the entries are non-negative -- for a connected non-negative matrix Perron-Frobenius guarantees one such sign exists.


    Formula: A x = lambda_max x; centrality is the principal eigenvector

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative adjacency matrix.

    Returns
    -------
    RichResult
        ``centrality`` (max-scaled), ``unit``, ``eigenvalue``, ``n``.

    References
    ----------
    Bonacich (1972), Factoring and weighting approaches to status scores
    and clique identification, Journal of Mathematical Sociology
    2:113-120.  Paywalled; the measure is the principal eigenvector of
    the adjacency matrix, as it is universally described in the
    centrality literature (e.g. Bonacich 2000, Social Networks
    22:357-365, which restates his own definition).
    """
    A = C.mat(A)
    n = len(A)
    vals, vecs = C.eigsym(A)
    v = [vecs[i][0] for i in range(n)]
    if sum(v) < 0:
        v = [-x for x in v]
    mx = max(abs(x) for x in v)
    return RichResult(payload={
        "centrality": [x / mx for x in v] if mx > 0 else v,
        "unit": v, "eigenvalue": vals[0], "n": n,
        "method": "Eigenvector centrality (principal eigenvector)"})


eigenvector_centrality = eigcent


def cheatsheet():
    return "eigcen: Eigenvector centrality."
