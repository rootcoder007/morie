# morie.fn -- function file (rootcoder007/morie)
"""k-means clustering of compositions in centred log-ratio coordinates."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['compkmeans', 'compositional_kmeans']


def compkmeans(X, k, init=None, iters=20):
    """k-means clustering of compositions in centred log-ratio coordinates.

    Formula: minimise sum_r d_a(x_r, centre_{c(r)})^2, equivalently Euclidean k-means on clr(x_r); centres returned to the simplex by clr^-1

    Parameters
    ----------
    X : array-like, shape (n, D)
        One composition per row; all parts strictly positive.
    k : int
        Number of clusters.
    init : array-like of int or None
        1-based row indices of the k compositions used as starting centres; None uses the first k rows.  Supplied rather than drawn so the result is reproducible.
    iters : int
        Number of Lloyd iterations to run.  A fixed count, not a tolerance, so both language arms perform identically many updates.

    Returns
    -------
    RichResult
        ``assignment``, ``centres``, ``centres_clr``, ``inertia``, ``sizes``, ``k``, ``iters``, ``n``, ``D``.

    References
    ----------
    Aitchison, J. (1986), The Statistical Analysis of Compositional Data, Chapman and Hall, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The log-ratio algebra and the additive logistic normal law were taken instead from Mateu-Figueras, G., Pawlowsky-Glahn, V. and Egozcue, J. J., The normal distribution in some constrained sample spaces, arXiv:0802.2643 (published as SORT 37(1):29-56, 2013), Sects. 4.1 and 4.3, which attribute the law to Aitchison (1982, 1986); that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  The Aitchison distance is the Euclidean distance between clr coordinates, so k-means on the simplex under d_a is ordinary k-means run on clr(X); the cluster centre in clr space maps back through clr^-1 to the closed geometric mean of its members, which is the compositional centre.  Ties in the assignment step go to the LOWEST cluster index in both language arms, and an empty cluster keeps its previous centre; without both of those rules the two arms can diverge on data with exact ties.
    """
    Xm = C.mat(X)
    n = len(Xm)
    if n == 0:
        raise ValueError("X must have at least one composition")
    D = len(Xm[0])
    for row in Xm:
        if any(v <= 0.0 for v in row):
            raise ValueError("compositions must be strictly positive")
    k = int(k)
    if not 1 <= k <= n:
        raise ValueError("k must lie between 1 and the number of compositions")
    it = int(iters)
    if it < 0:
        raise ValueError("iters must be non-negative")
    Zc = []
    for row in Xm:
        lg = sum(math.log(v) for v in row) / D
        Zc.append([math.log(v) - lg for v in row])
    if init is None:
        seed = list(range(1, k + 1))
    else:
        seed = [int(v) for v in init]
    if len(seed) != k or len(set(seed)) != k or any(not 1 <= i <= n for i in seed):
        raise ValueError("init must be k distinct 1-based row indices")
    Cen = [list(Zc[i - 1]) for i in seed]
    asg = [0] * n
    for _ in range(it):
        for r in range(n):
            best, bd = 0, None
            for c in range(k):
                dd = sum((Zc[r][j] - Cen[c][j]) ** 2 for j in range(D))
                if bd is None or dd < bd:
                    best, bd = c, dd
            asg[r] = best
        for c in range(k):
            mem = [r for r in range(n) if asg[r] == c]
            if mem:
                Cen[c] = [sum(Zc[r][j] for r in mem) / len(mem) for j in range(D)]
    inertia = 0.0
    for r in range(n):
        inertia += sum((Zc[r][j] - Cen[asg[r]][j]) ** 2 for j in range(D))
    out = []
    for c in range(k):
        m = max(Cen[c])
        e = [math.exp(val - m) for val in Cen[c]]
        s = sum(e)
        out.append([val / s for val in e])
    return RichResult(payload={
        "assignment": [a + 1 for a in asg], "centres": out, "centres_clr": Cen,
        "inertia": inertia, "sizes": [sum(1 for a in asg if a == c) for c in range(k)],
        "k": k, "iters": it, "n": n, "D": D,
        "method": "Compositional k-means in clr coordinates"})


compositional_kmeans = compkmeans


def cheatsheet():
    return 'aitclu: k-means clustering of compositions in centred log-ratio coordinates.'
