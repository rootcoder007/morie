# morie.fn -- function file (rootcoder007/morie)
"""PERMANOVA pseudo-F from a distance matrix."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["permanova", "compositional_permanova"]


def permanova(X, group, aitchison=True):
    """One-way PERMANOVA pseudo-F on Aitchison (or Euclidean) distances.

    Anderson's non-parametric MANOVA works entirely from the N x N matrix
    of pairwise distances.  With N units in a groups of sizes n_g,

        SS_T = (1/N) sum_{i<j} d_ij^2
        SS_W = sum_{i<j} d_ij^2 eps_ij / n_{g(i)},  eps_ij = 1 iff same group
        SS_A = SS_T - SS_W
        F    = (SS_A / (a - 1)) / (SS_W / (N - a))

    For compositions the natural distance is Aitchison's, which is the
    Euclidean distance between centred log-ratio scores; that is the
    default here.  No permutation is performed -- the statistic is
    closed-form, and permuting would make the result depend on a random
    stream.

    Parameters
    ----------
    X : array-like, shape (N, D)
        Rows are compositions (strictly positive) when ``aitchison`` is
        true, otherwise plain numeric vectors.
    group : array-like
        Group label per row; any hashable labels.
    aitchison : bool
        Take clr coordinates before measuring Euclidean distance.

    Returns
    -------
    RichResult
        ``F``, ``SSA``, ``SSW``, ``SST``, ``df1``, ``df2``, ``N``, ``a``,
        ``sizes``.

    References
    ----------
    Anderson, M. J. (2001), "A new method for non-parametric multivariate
    analysis of variance", Austral Ecology 26(1), 32-46, whose Equations
    (3)-(5) give SS_T, SS_W and the pseudo-F above from the interpoint
    distance matrix.  Standard published form; the Austral Ecology
    article is paywalled and the download attempted for this
    implementation returned a stub, so the article was not read.  The
    Aitchison distance is the Euclidean distance of clr scores
    (Aitchison 1986).
    """
    M = C.mat(X)
    N, D = len(M), len(M[0])
    g = list(group)
    if len(g) != N:
        raise ValueError("group must have one label per row")
    if N < 3:
        raise ValueError("need at least three units")
    if aitchison:
        if any(v <= 0.0 for r in M for v in r):
            raise ValueError("compositions must be strictly positive")
        Y = []
        for r in M:
            lg = [math.log(v) for v in r]
            gm = sum(lg) / D
            Y.append([v - gm for v in lg])
    else:
        Y = M
    labs = []
    for v in g:
        if v not in labs:
            labs.append(v)
    a = len(labs)
    if a < 2:
        raise ValueError("need at least two groups")
    size = {L: sum(1 for v in g if v == L) for L in labs}
    sst = 0.0
    ssw = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d2 = sum((Y[i][k] - Y[j][k]) ** 2 for k in range(D))
            sst += d2
            if g[i] == g[j]:
                ssw += d2 / size[g[i]]
    sst /= N
    ssa = sst - ssw
    df1, df2 = a - 1, N - a
    return RichResult(payload={
        "F": (ssa / df1) / (ssw / df2) if ssw > 0.0 else float("inf"),
        "SSA": ssa, "SSW": ssw, "SST": sst, "df1": df1, "df2": df2,
        "N": N, "a": a, "sizes": [size[L] for L in labs],
        "method": "PERMANOVA pseudo-F on Aitchison distances (Anderson 2001)"})


compositional_permanova = permanova


def cheatsheet():
    return "aitprm: PERMANOVA pseudo-F from a distance matrix."
