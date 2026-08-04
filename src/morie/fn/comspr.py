# morie.fn -- function file (rootcoder007/morie)
"""Spectral clustering."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["specclus", "spectral_clustering"]


def specclus(W, k=2, normalized=True, max_iter=50):
    """Spectral clustering: embed by the bottom eigenvectors, then k-means.

    The eigenvectors are the SMALLEST ones, not the largest -- a
    Laplacian is a penalty, so the informative directions are the ones
    it penalises least.  Under the symmetric normalization the
    embedding rows are additionally scaled to unit length, which is
    what makes the ideal-case clusters land on orthogonal points of
    the sphere.

    Eigenvector signs are fixed on the largest-magnitude entry, and the
    k-means is initialised at the FIRST k embedded rows with a fixed
    sweep budget, so the two language arms return the same labels.
    Under a repeated eigenvalue the individual eigenvectors are not
    unique, so the labels can still change if the graph is exactly
    symmetric; that is a property of the problem, not of the code.

    Formula: L = T - W (or L_sym = T^-1/2 L T^-1/2);
             U = the k eigenvectors of smallest eigenvalue;
             normalized: rows of U scaled to unit norm;
             cluster the rows of U by k-means

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative similarity matrix.
    k : int
        Number of clusters, 2 <= k <= n.
    normalized : bool
        Use the symmetric normalized Laplacian and row-normalize the
        embedding (the Ng-Jordan-Weiss variant); otherwise the
        unnormalized algorithm of the tutorial.
    max_iter : int
        k-means sweep budget.

    Returns
    -------
    RichResult
        ``cluster`` (one-based), ``embedding`` (n x k), ``values``,
        ``centers``, ``tot_withinss``, ``iterations``, ``n``, ``k``.

    References
    ----------
    von Luxburg (2007), A Tutorial on Spectral Clustering, Statistics
    and Computing 17(4), 395-416, whose boxed "Unnormalized spectral
    clustering" takes the similarity matrix W, computes the
    unnormalized Laplacian L, computes the first k eigenvectors
    u_1, ..., u_k of L, forms U in R^{n x k} with those as columns,
    lets y_i be the i-th row of U, and clusters the y_i with k-means;
    the normalized variant additionally normalizes the rows of U.
    Fetched from arXiv:0711.0189.  Lloyd (1982), IEEE Transactions on
    Information Theory 28(2), 129-137, for the k-means step.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    k = int(k)
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")
    d = [sum(W[i]) for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    if normalized:
        s = [0.0 if d[i] == 0.0 else d[i] ** -0.5 for i in range(n)]
        L = [[s[i] * L[i][j] * s[j] for j in range(n)] for i in range(n)]
    vals, vecs = C.eigsym(L)
    order = list(reversed(range(n)))
    lam = [vals[i] for i in order]
    U = [[vecs[r][order[j]] for j in range(k)] for r in range(n)]
    if normalized:
        for r in range(n):
            nr = math.sqrt(sum(v * v for v in U[r]))
            if nr > 0:
                U[r] = [v / nr for v in U[r]]
    cen = [list(U[i]) for i in range(k)]
    # -1, not 0: with 0-based labels a first-sweep assignment to cluster 0
    # would not register as a move and the loop would stop a sweep early,
    # which is exactly where the R arm (1-based labels) disagreed.
    lab = [-1] * n
    it = 0
    for it in range(1, int(max_iter) + 1):
        moved = False
        for i in range(n):
            best = 0
            bd = None
            for c in range(k):
                dd = sum((U[i][j] - cen[c][j]) ** 2 for j in range(k))
                if bd is None or dd < bd:
                    bd = dd
                    best = c
            if lab[i] != best:
                lab[i] = best
                moved = True
        for c in range(k):
            mem = [i for i in range(n) if lab[i] == c]
            if mem:
                cen[c] = [sum(U[i][j] for i in mem) / len(mem)
                          for j in range(k)]
        if not moved:
            break
    wss = 0.0
    for i in range(n):
        wss += sum((U[i][j] - cen[lab[i]][j]) ** 2 for j in range(k))
    return RichResult(payload={
        "cluster": [v + 1 for v in lab], "embedding": U,
        "values": lam[:k], "centers": cen, "tot_withinss": wss,
        "iterations": float(it), "n": float(n), "k": float(k),
        "method": "Spectral clustering, von Luxburg (2007)"})


spectral_clustering = specclus


def cheatsheet():
    return "comspr: bottom-k Laplacian eigenvectors, row-normalized, then k-means"
