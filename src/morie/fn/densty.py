# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Network density."""

import numpy as np

from ._richresult import RichResult

__all__ = ["density"]


def density(G):
    """
    Density of an undirected simple graph.

    Formula: Delta = |E| / C(n, 2), the fraction of possible edges
    present. ``G`` is an n x n adjacency matrix; it must be square,
    symmetric, hollow (zero diagonal — no self-loops), and binary.
    Directed or weighted structures are refused rather than silently
    reinterpreted.

    Parameters
    ----------
    G : array-like, shape (n, n)
        Binary symmetric adjacency matrix, n >= 2.

    Returns
    -------
    result : dict
        Keys: estimate (density), n_edges, n_possible, n, method.

    References
    ----------
    Wasserman & Faust (1994), Ch 4.

    Examples
    --------
    Triangle: 3 of 3 possible edges.

    >>> K3 = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    >>> density(K3)["estimate"]
    1.0
    >>> path = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    >>> out = density(path)
    >>> round(out["estimate"], 12)
    0.666666666667
    >>> out["n_edges"]
    2
    >>> density([[0, 1], [0, 0]])
    Traceback (most recent call last):
        ...
    ValueError: the adjacency matrix must be symmetric (undirected graph).
    """
    A = np.asarray(G, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"the adjacency matrix must be square; got shape {A.shape}.")
    n = A.shape[0]
    if n < 2:
        raise ValueError("density needs at least 2 vertices.")
    if not np.array_equal(A, A.T):
        raise ValueError("the adjacency matrix must be symmetric (undirected graph).")
    if np.any(np.diag(A) != 0):
        raise ValueError("self-loops are not allowed (nonzero diagonal).")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("the adjacency matrix must be binary.")
    edges = int(np.sum(A) // 2)
    possible = n * (n - 1) // 2
    return RichResult(payload={
        "estimate": float(edges / possible), "n_edges": edges,
        "n_possible": int(possible), "n": int(n),
        "method": "density |E| / C(n,2), undirected simple graph"})


def cheatsheet():
    return "densty: |E|/C(n,2); square+symmetric+hollow+binary enforced"
