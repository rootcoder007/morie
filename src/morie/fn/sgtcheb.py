# morie.fn -- function file (rootcoder007/morie)
"""Cheeger ratio of a vertex subset and the Cheeger bounds on the spectral gap."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['cheegbnd', 'sgt_cheeger_bound']


def cheegbnd(Adj, S):
    """Cheeger ratio of a vertex subset and the Cheeger bounds on the spectral gap.

    Formula: h_S = |dS| / min{vol(S), vol(G) - vol(S)};  2 h >= lambda_G >= h^2 / 2

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
    S : array-like of int
        1-based indices of the vertices in the subset; must be a proper non-empty subset.

    Returns
    -------
    RichResult
        ``cheeger_ratio``, ``boundary``, ``vol_S``, ``vol_complement``, ``vol_G``, ``upper_bound``, ``lower_bound``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: dS = {{u, v} in E : u in S, v not in S}, h_S = |dS| / min{vol(S), vol(G) - vol(S)}, and the Cheeger constant h_G is the minimum of h_S over all subsets.  Sect. 1 states the Cheeger inequality 2 h_G >= lambda_G >= h_G^2 / 2 for a connected graph.  Only ONE subset is supplied here, so ``cheeger_ratio`` is h_S and not the constant h_G; the two bounds are reported as what h_S would give if this subset were the minimising one, and are labelled accordingly rather than presented as a proven bracket on lambda_G.  |dS| is the total weight crossing the cut, which is the edge count for an unweighted graph.
    """
    A = C.mat(Adj)
    n = len(A)
    if n == 0 or len(A[0]) != n:
        raise ValueError("Adj must be a non-empty square matrix")
    for i in range(n):
        for j in range(n):
            if A[i][j] < 0.0:
                raise ValueError("edge weights must be non-negative")
            if abs(A[i][j] - A[j][i]) > 0.0:
                raise ValueError("Adj must be symmetric")
    d = [sum(A[i]) for i in range(n)]

    idx = sorted({int(v) for v in S})
    if not idx:
        raise ValueError("S must be non-empty")
    if any(not 1 <= i <= n for i in idx):
        raise ValueError("S must contain 1-based vertex indices")
    if len(idx) == n:
        raise ValueError("S must be a proper subset")
    inS = [False] * n
    for i in idx:
        inS[i - 1] = True
    cut = sum(A[i][j] for i in range(n) for j in range(n) if inS[i] and not inS[j])
    volS = sum(d[i] for i in range(n) if inS[i])
    volG = sum(d)
    volC = volG - volS
    den = min(volS, volC)
    if den <= 0.0:
        raise ValueError("both sides of the cut must have positive volume")
    h = cut / den
    return RichResult(payload={
        "cheeger_ratio": h, "boundary": cut, "vol_S": volS,
        "vol_complement": volC, "vol_G": volG,
        "upper_bound": 2.0 * h, "lower_bound": h * h / 2.0, "n": n,
        "method": "Cheeger ratio and Cheeger bounds"})


sgt_cheeger_bound = cheegbnd


def cheatsheet():
    return 'sgtcheb: Cheeger ratio of a vertex subset and the Cheeger bounds on the spectral gap.'
