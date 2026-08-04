# morie.fn -- function file (rootcoder007/morie)
"""Cheeger constant and the Cheeger bounds."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["cheeger", "sgt_cheeger_bound"]


def cheeger(W, max_n=20):
    """Exact Cheeger constant, with the eigenvalue bounds it satisfies.

    The Cheeger constant is computed by exhaustive enumeration of the
    2^(n-1) - 1 vertex bipartitions, so it is exact rather than a
    relaxation, and the returned bounds are then a genuine check on
    ``sgtspc`` rather than a definition of it.  Enumeration is
    exponential, hence the ``max_n`` guard: this is a teaching and
    testing routine, not a partitioner for large graphs.

    Formula: h_G(S) = |E(S, Sbar)| / min(vol S, vol Sbar),
             h_G = min_S h_G(S);
             Cheeger inequality  2 h_G >= lambda_1 > h_G^2 / 2,
             and the sharper  lambda_1 >= 1 - sqrt(1 - h_G^2)

    Parameters
    ----------
    W : array-like, shape (n, n)
        Symmetric non-negative weight matrix, connected.
    max_n : int
        Refuse to enumerate beyond this many vertices.

    Returns
    -------
    RichResult
        ``h``, ``argmin`` (one-based vertices of the minimising S),
        ``cut``, ``vol_S``, ``vol_complement``, ``lambda1``,
        ``upper_bound`` (2h), ``lower_bound`` (h^2/2),
        ``lower_bound_sharp`` (1 - sqrt(1 - h^2)), ``n``.

    References
    ----------
    Chung (1997), Spectral Graph Theory, CBMS 92, Section 2.2,
    equations (2.1) and (2.2) for h_G(S) and h_G; Theorem 2.2 for
    2 h_G >= lambda_1 > h_G^2 / 2; Theorem 2.3 for
    lambda_1 >= 1 - sqrt(1 - h_G^2).  Fetched from the author's own
    copy of the chapter.
    """
    W = C.mat(W)
    n = len(W)
    if any(len(r) != n for r in W):
        raise ValueError("W must be square")
    if n > int(max_n):
        raise ValueError("exact enumeration refused above max_n vertices")
    if n < 2:
        raise ValueError("the Cheeger constant needs at least two vertices")
    d = [sum(W[i]) for i in range(n)]
    vol = sum(d)
    best = None
    arg = []
    cut = 0.0
    vs = 0.0
    # Enumerate subsets containing vertex 1 only: S and its complement
    # give the same h, so half the lattice is redundant.
    for mask in range(1, 1 << (n - 1)):
        S = [0] + [i for i in range(1, n) if mask >> (i - 1) & 1]
        inS = [False] * n
        for i in S:
            inS[i] = True
        volS = sum(d[i] for i in S)
        if volS == 0 or volS == vol:
            continue
        e = 0.0
        for i in S:
            for j in range(n):
                if not inS[j]:
                    e += W[i][j]
        h = e / min(volS, vol - volS)
        if best is None or h < best:
            best = h
            arg = [i + 1 for i in S]
            cut = e
            vs = volS
    s = [0.0 if d[i] == 0.0 else d[i] ** -0.5 for i in range(n)]
    L = [[(d[i] - W[i][i]) if i == j else -W[i][j] for j in range(n)]
         for i in range(n)]
    Lc = [[s[i] * L[i][j] * s[j] for j in range(n)] for i in range(n)]
    vals = list(reversed(C.eigsym(Lc)[0]))
    nz = [v for v in vals if v > 1e-10]
    lam1 = nz[0] if nz else 0.0
    sharp = 1.0 - math.sqrt(max(0.0, 1.0 - best * best)) if best <= 1 else 1.0
    return RichResult(payload={
        "h": best, "argmin": arg, "cut": cut, "vol_S": vs,
        "vol_complement": vol - vs, "lambda1": lam1,
        "upper_bound": 2.0 * best, "lower_bound": best * best / 2.0,
        "lower_bound_sharp": sharp, "n": n,
        "method": "Cheeger constant with Chung Theorems 2.2 and 2.3"})


sgt_cheeger_bound = cheeger


def cheatsheet():
    return "sgtcheb: h_G = min |E(S,Sbar)|/min(vol S, vol Sbar); 2h >= l1 > h^2/2"
