# morie.fn -- function file (rootcoder007/morie)
"""Cheeger constant by a sweep over the Fiedler vector."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['cheeger', 'sgt_cheeger_constant']


def cheeger(A):
    """Cheeger constant by a sweep over the Fiedler vector.

    The exact constant is an NP-hard minimisation over all vertex subsets. The sweep evaluates only the n-1 prefixes of the ordering induced by the second eigenvector of the Laplacian, which is what Cheeger's inequality justifies: the value found is an upper bound on h(G), and never below it. It is returned as ``sweep_min`` rather than as h(G), together with the spectral lower bound lambda_2/2, so the gap the method leaves is visible instead of implied. The eigenvector must come from the NORMALISED Laplacian, and the sweep must order by D^-1/2 v: pairing a degree-weighted conductance with the combinatorial Laplacian D - A looks almost right and produces a lower bound that the true constant falls below.


    Formula: h(G) = min_S |boundary(S)| / min(vol S, vol S^c); lambda_2/2 <= h(G) <= sqrt(2 lambda_2) for the NORMALISED Laplacian I - D^-1/2 A D^-1/2

    Parameters
    ----------
    A : array-like, shape (n, n)
        Symmetric non-negative adjacency matrix.

    Returns
    -------
    RichResult
        ``sweep_min``, ``lower_bound`` (lambda_2/2), ``upper_bound`` (sqrt(2 lambda_2)), ``lambda2``, ``cut_set``, ``fiedler``, ``n``.

    References
    ----------
    Cheeger (1970), A lower bound for the smallest eigenvalue of the
    Laplacian, in Problems in Analysis; Chung (1997), Spectral Graph
    Theory, AMS.  Neither is held locally; the conductance definition
    and the sweep-cut construction are standard published results.  The
    sweep value is checked against exhaustive enumeration over all
    subsets in the batch's anchor file.
    """
    A = C.mat(A)
    n = len(A)
    deg = [sum(A[i][j] for j in range(n) if j != i) for i in range(n)]
    if any(d <= 0 for d in deg):
        raise ValueError("isolated vertices: conductance is undefined")
    ds = [1.0 / math.sqrt(d) for d in deg]
    L = [[(1.0 if i == j else 0.0)
          - (ds[i] * A[i][j] * ds[j] if i != j else 0.0)
          for j in range(n)] for i in range(n)]
    vals, vecs = C.eigsym(L)
    lam2 = vals[n - 2]
    f = [vecs[i][n - 2] * ds[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: (f[i], i))
    total = sum(deg)
    best, bestset = float("inf"), []
    for k in range(1, n):
        Sset = order[:k]
        inS = [False] * n
        for v in Sset:
            inS[v] = True
        cut = sum(A[i][j] for i in range(n) for j in range(n)
                  if i != j and inS[i] and not inS[j])
        vol = sum(deg[v] for v in Sset)
        den = min(vol, total - vol)
        if den > 0:
            val = cut / den
            if val < best:
                best, bestset = val, sorted(Sset)
    return RichResult(payload={
        "sweep_min": best, "lower_bound": lam2 / 2.0,
        "upper_bound": math.sqrt(2.0 * lam2) if lam2 > 0 else 0.0,
        "lambda2": lam2, "cut_set": bestset, "fiedler": f, "n": n,
        "method": "Cheeger constant (Fiedler sweep upper bound)"})


sgt_cheeger_constant = cheeger


def cheatsheet():
    return "sgtcheeg: Cheeger constant by a sweep over the Fiedler vector."
