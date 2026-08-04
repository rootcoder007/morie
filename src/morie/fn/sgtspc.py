# morie.fn -- function file (rootcoder007/morie)
"""Spectrum and spectral gap of the normalized Laplacian."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['glapspec', 'sgt_spectrum', 'sgtspectrum']


def glapspec(Adj, tol=1e-9):
    """Spectrum and spectral gap of the normalized Laplacian.

    Formula: eigenvalues of L = I - D^(-1/2) A D^(-1/2);  lambda_G = least nonzero eigenvalue

    Parameters
    ----------
    Adj : array-like, shape (n, n)
        Symmetric adjacency matrix with a zero diagonal; entries may be 0/1 or non-negative edge weights.
    tol : float
        Eigenvalues no larger than tol in absolute value count as zero when the spectral gap is picked out.

    Returns
    -------
    RichResult
        ``eigenvalues``, ``spectral_gap``, ``n_zero``, ``lambda_max``, ``n``.

    References
    ----------
    Chung, F. (1997), Spectral Graph Theory, CBMS Regional Conference Series in Mathematics 92, American Mathematical Society, is this shelf's primary book and is NOT in the reference library, so it could not be read.  The conventions below were taken instead from the author's own survey, Chung, F., Four proofs for the Cheeger inequality and graph partition algorithms, Proceedings of ICCM 2007 Vol. II pp. 1-4, Sect. 2 (Preliminaries), which restates them in her notation; that paper was FETCHED and is archived in the reference library with a row in EXTERNAL_SOURCES.md.  Sect. 2: the spectral gap lambda_G is the least NONZERO eigenvalue of the normalized Laplacian.  Its eigenvalues lie in [0, 2] and the multiplicity of the eigenvalue 0 is the number of connected components, so ``n_zero`` counts components.  Eigenvalues are returned in increasing order.
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

    s = [0.0 if d[i] == 0.0 else 1.0 / math.sqrt(d[i]) for i in range(n)]
    L = [[(1.0 if i == j else 0.0) - s[i] * A[i][j] * s[j] for j in range(n)]
         for i in range(n)]
    vals, _ = C.eigsym(L)
    lam = sorted(vals)
    tol = float(tol)
    nz = sum(1 for v in lam if abs(v) <= tol)
    gap = next((v for v in lam if abs(v) > tol), float("nan"))
    return RichResult(payload={
        "eigenvalues": lam, "spectral_gap": gap, "n_zero": nz,
        "lambda_max": lam[-1], "n": n,
        "method": "Normalized Laplacian spectrum and spectral gap"})


sgt_spectrum = glapspec
sgtspectrum = glapspec


def cheatsheet():
    return 'sgtspc: Spectrum and spectral gap of the normalized Laplacian.'
