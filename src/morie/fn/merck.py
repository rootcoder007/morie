# morie.fn -- function file (rootcoder007/morie)
"""Mercer check: is a Gram matrix a valid kernel matrix?"""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['mercerchk', 'mercer_theorem', 'mercertheorem']


def mercerchk(K, tol=1e-9):
    """Mercer check: is a Gram matrix a valid kernel matrix?

    Formula: K is a kernel iff the Gram matrix is symmetric positive semi-definite: min eigenvalue(K) >= 0

    Parameters
    ----------
    K : array-like, shape (n, n)
        Candidate Gram matrix.
    tol : float
        Negative eigenvalues larger in size than tol fail the check.

    Returns
    -------
    RichResult
        ``is_kernel``, ``min_eigenvalue``, ``eigenvalues``, ``symmetry_gap``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Sect. 8.2.1 p. 255, property 2 and property 3: the Gram matrix K with entries K(x_i, x_j) must be positive semi-definite for every choice of x_1, ..., x_n, and the book states that Mercer's theorem -- the integral condition on square-integrable g -- is an equivalent formulation of that finitely positive semi-definite property.  The finite check is therefore the one implemented.  Read from the chapter PDF, not recalled.
    """
    Km = G._mat(K)
    n = len(Km)
    if n == 0 or len(Km[0]) != n:
        raise ValueError("K must be a non-empty square matrix")
    gap = max(abs(Km[i][j] - Km[j][i]) for i in range(n) for j in range(n))
    ok, lam = G.is_positive_semidefinite(Km, tol=float(tol))
    return RichResult(payload={
        "is_kernel": bool(ok), "min_eigenvalue": min(lam), "eigenvalues": lam,
        "symmetry_gap": gap, "n": n,
        "method": "Mercer / positive semi-definiteness check, MVSML Sect. 8.2.1"})


mercer_theorem = mercerchk
mercertheorem = mercerchk


def cheatsheet():
    return 'merck: Mercer check: is a Gram matrix a valid kernel matrix?'
