# morie.fn -- function file (rootcoder007/morie)
"""Spectral clustering on a similarity matrix."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import eye, sqmat, topeigs

__all__ = [
    "spectral_clustering",
    "specclust",
]


def spectral_clustering(a, k=2):
    """Cluster the nodes of a similarity graph through its Laplacian.

    NOT IN SCHABENBERGER & GOTWAY -- a fixed-string search for "spectral
    clustering" and "principal component" returns nothing. The source is
    Ng, A. Y., Jordan, M. I. & Weiss, Y. (2001), "On spectral clustering:
    analysis and an algorithm", NIPS 14 -- named from the general
    literature and NOT verified against a PDF in this corpus.

        L_sym = I - D^{-1/2} A D^{-1/2}

    and the clustering lives in the eigenvectors belonging to the SMALLEST
    eigenvalues of L_sym. Because power iteration finds the LARGEST, the
    iteration runs on ``2 I - L_sym``, whose largest eigenvalues are
    L_sym's smallest; the eigenvalues are mapped back before they are
    reported. Running power iteration on L_sym directly and taking the top
    vectors is the standard way to get this exactly backwards.

    A node with zero degree has no D^{-1/2}, and rather than silently
    substituting zero the function raises: an isolated node does not
    belong to any cluster and the caller should decide what to do with it.

    The partition is taken from the second-smallest eigenvector (the
    Fiedler vector): sign split for k = 2, and for k > 2 a one-dimensional
    k-means on that coordinate whose centres START at k evenly spaced
    ORDER STATISTICS rather than random points. That start is
    deterministic, which is what lets the two language arms agree; it is
    also the reason no `seed` argument exists.

    Parameters
    ----------
    a : (n, n) array-like
        Symmetric, non-negative similarity matrix, zero diagonal.
    k : int
        Number of clusters, 2 <= k <= n.

    Returns
    -------
    RichResult
        ``labels``, ``sizes``, ``eigenvalues``, ``fiedler``,
        ``degree``, ``n``, ``method``.
    """
    w = sqmat(a, None, "a")
    n = len(w)
    k = int(k)
    if k < 2 or k > n:
        raise ValueError("`k` must lie between 2 and the number of nodes")
    for i in range(n):
        if w[i][i] != 0.0:
            raise ValueError("`a` must have a zero diagonal")
        for j in range(n):
            if w[i][j] < 0:
                raise ValueError("`a` must be non-negative")
            if abs(w[i][j] - w[j][i]) > 1e-12:
                raise ValueError("`a` must be symmetric")
    deg = [fsum(row) for row in w]
    for i in range(n):
        if deg[i] <= 0:
            raise ValueError("node %d has degree 0; an isolated node "
                             "belongs to no cluster" % i)
    ds = [1.0 / sqrt(t) for t in deg]

    lsym = eye(n)
    for i in range(n):
        for j in range(n):
            lsym[i][j] = lsym[i][j] - ds[i] * w[i][j] * ds[j]
    shifted = [[(2.0 if i == j else 0.0) - lsym[i][j] for j in range(n)]
               for i in range(n)]
    vals, vecs = topeigs(shifted, min(k, n))
    eig = [2.0 - t for t in vals]
    fied = vecs[1] if len(vecs) > 1 else vecs[0]

    if k == 2:
        labels = [0.0 if t < 0 else 1.0 for t in fied]
    else:
        srt = sorted(fied)
        cen = [srt[int(round((n - 1) * (c + 0.5) / k))] for c in range(k)]
        labels = [0.0] * n
        for _ in range(50):
            for i in range(n):
                best = 0
                for c in range(k):
                    if abs(fied[i] - cen[c]) < abs(fied[i] - cen[best]):
                        best = c
                labels[i] = float(best)
            for c in range(k):
                mem = [fied[i] for i in range(n) if labels[i] == float(c)]
                if mem:
                    cen[c] = fsum(mem) / len(mem)

    sizes = [float(len([t for t in labels if t == float(c)]))
             for c in range(k)]

    return RichResult(payload={
        "labels": labels,
        "sizes": sizes,
        "eigenvalues": eig,
        "fiedler": fied,
        "degree": deg,
        "smallest_eigenvalues_not_largest": True,
        "k": float(k),
        "n": n,
        "method": ("Normalized spectral clustering (Ng, Jordan & Weiss "
                   "2001) with deterministic order-statistic starts; NOT "
                   "in Schabenberger & Gotway"),
    })


def cheatsheet():
    return "specClust: normalized spectral clustering"


# compact alias per ledger/NAMING.md
specclust = spectral_clustering
