# morie.fn -- function file (rootcoder007/morie)
"""Isomap -- Tenenbaum, de Silva & Langford (2000), ESL Sec 14.9."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_isomap"]


def esl_isomap(X, k=2, neighbors=5):
    r"""Embed data by classical MDS on geodesic distances.

    Isomap replaces the straight-line distance with the shortest path along a
    ``neighbors``-nearest-neighbour graph, approximating the geodesic on the
    manifold, then applies classical MDS:

    .. math::
        B = -\tfrac12 H D^{(g)2} H, \qquad H = I - \tfrac1n \mathbf{1}\mathbf{1}^\top,

    with the embedding read off the top eigenvectors of :math:`B`. For the
    Swiss roll this is the whole point: two points on facing sheets are close
    in space and far along the surface, and only the graph distance knows the
    difference.

    The neighbourhood size is the method's weak spot. Too small and the graph
    disconnects (no finite path exists, and the embedding of a disconnected
    graph is not defined); too large and it "short-circuits" across a fold,
    silently reintroducing the Euclidean distance it was built to avoid.
    Disconnection is detected and raised rather than filled in.

    Parameters
    ----------
    X : array-like
        Data ``(n, p)``.
    k : int
        Embedding dimension.
    neighbors : int
        Neighbourhood size for the graph, at least 1.

    Returns
    -------
    RichResult
        ``embedding`` ``(n, k)``, ``eigenvalues``, ``geodesic`` distances,
        ``residual_variance``, ``n_components``.

    References
    ----------
    Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global
        geometric framework for nonlinear dimensionality reduction.
        *Science*, 290(5500), 2319-2323.
    Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
        Statistical Learning* (2nd ed.). Springer.

    Examples
    --------
    On a Swiss roll the embedding recovers the roll's own arclength
    parameter, which a straight-line method cannot.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, 400)
    >>> h = rng.uniform(0, 10, 400)
    >>> X = np.column_stack([t * np.cos(t), h, t * np.sin(t)])
    >>> emb = esl_isomap(X, k=2, neighbors=8)["embedding"]
    >>> bool(abs(np.corrcoef(emb[:, 0], t)[0, 1]) > 0.9)
    True

    Geodesic distance exceeds straight-line distance, because a path along
    the surface can never be shorter than the chord.

    >>> r = esl_isomap(X, k=2, neighbors=8)
    >>> D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    >>> bool(np.all(r["geodesic"] >= D - 1e-9))
    True

    Too few neighbours disconnects the graph, and that is raised rather
    than papered over.

    >>> esl_isomap(np.array([[0.0, 0.0], [0.0, 1.0], [50.0, 50.0], [50.0, 51.0]]),
    ...            k=1, neighbors=1)
    Traceback (most recent call last):
        ...
    ValueError: the 1-nearest-neighbour graph has 2 disconnected components; raise `neighbors`
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    k, neighbors = int(k), int(neighbors)
    if not 1 <= k < n:
        raise ValueError(f"k must be between 1 and {n - 1}")
    if not 1 <= neighbors < n:
        raise ValueError(f"neighbors must be between 1 and {n - 1}")

    D = np.sqrt(np.maximum(((X[:, None] - X[None]) ** 2).sum(-1), 0.0))
    G = np.full((n, n), np.inf)
    np.fill_diagonal(G, 0.0)
    idx = np.argsort(D, axis=1)[:, 1: neighbors + 1]
    for i in range(n):
        G[i, idx[i]] = D[i, idx[i]]
    G = np.minimum(G, G.T)                       # symmetrise the graph

    for m in range(n):                           # Floyd-Warshall
        G = np.minimum(G, G[:, m][:, None] + G[m][None, :])

    if not np.all(np.isfinite(G)):
        n_comp = _count_components(np.isfinite(G))
        raise ValueError(
            f"the {neighbors}-nearest-neighbour graph has {n_comp} disconnected "
            "components; raise `neighbors`"
        )

    Hc = np.eye(n) - 1.0 / n
    B = -0.5 * Hc @ (G**2) @ Hc
    w, V = np.linalg.eigh((B + B.T) / 2)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = np.clip(w[:k], 0, None)
    emb = V[:, :k] * np.sqrt(pos)

    Dg = np.sqrt(np.maximum(((emb[:, None] - emb[None]) ** 2).sum(-1), 0.0))
    iu = np.triu_indices(n, 1)
    rv = float(1 - np.corrcoef(G[iu], Dg[iu])[0, 1] ** 2)
    return RichResult(
        title="Isomap",
        summary_lines=[("n", n), ("k", k), ("neighbors", neighbors),
                       ("residual variance", rv)],
        payload={
            "embedding": emb, "eigenvalues": w, "geodesic": G,
            "residual_variance": rv, "n_components": 1,
            "neighbors": neighbors,
            "method": "esl_isomap",
        },
    )


def _count_components(adj):
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for s in range(n):
        if seen[s]:
            continue
        comps += 1
        stack = [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u] & ~seen):
                seen[v] = True
                stack.append(int(v))
    return comps


def cheatsheet():
    return "eslism: MDS on graph geodesics; too-few neighbours disconnects (raises), too many short-circuit folds"
