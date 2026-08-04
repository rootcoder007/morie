# morie.fn -- function file (rootcoder007/morie)
"""Cholesky re-parameterization that makes GBLUP an ordinary mixed model."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['gblupeq', 'gblup_equivalence']


def gblupeq(Z, G, sigma2_g):
    """Cholesky re-parameterization that makes GBLUP an ordinary mixed model.

    Formula: G = L L';  Zstar = Z L;  then Y = X beta + Zstar ustar + e with ustar ~ N(0, sigma2_g I) has the same marginal variance as u ~ N(0, sigma2_g G)

    Parameters
    ----------
    Z : array-like, shape (n, q)
        Design matrix of lines.
    G : array-like, shape (q, q)
        Genomic relationship matrix; must be positive definite.
    sigma2_g : float
        Genomic variance component.

    Returns
    -------
    RichResult
        ``Zstar``, ``L``, ``V_original``, ``V_reparameterized``, ``max_gap``, ``n``, ``q``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, p. 46: the GBLUP model can be expressed equivalently as Y = X beta + Z* u + e with Z* = Z L', G = L'L the Cholesky decomposition of G, and u ~ N(0, sigma2_g I_q).  The book's L is the upper factor returned by R's chol(); this implementation uses the lower factor L with G = L L' and Z* = Z L in BOTH language arms, which gives the identical marginal variance sigma2_g Z G Z' -- ``max_gap`` reports the largest entry-wise difference between the two marginal variances, and is zero up to rounding.  Read from the chapter PDF, not recalled.
    """
    Zm = C.mat(Z)
    Gm = C.mat(G)
    s2 = float(sigma2_g)
    n, q = len(Zm), len(Zm[0])
    if len(Gm) != q or len(Gm[0]) != q:
        raise ValueError("G must be q by q with q the number of columns of Z")
    if s2 < 0.0:
        raise ValueError("sigma2_g must be non-negative")
    L = C.chol(Gm)
    Zs = C.matmul(Zm, L)
    V0 = C.matmul(C.matmul(Zm, Gm), C.transpose(Zm))
    V1 = C.matmul(Zs, C.transpose(Zs))
    V0 = [[s2 * V0[i][j] for j in range(n)] for i in range(n)]
    V1 = [[s2 * V1[i][j] for j in range(n)] for i in range(n)]
    gap = max(abs(V0[i][j] - V1[i][j]) for i in range(n) for j in range(n))
    return RichResult(payload={
        "Zstar": Zs, "L": L, "V_original": V0, "V_reparameterized": V1,
        "max_gap": gap, "n": n, "q": q,
        "method": "GBLUP Cholesky re-parameterization, MVSML Chap. 2 p. 46"})


gblup_equivalence = gblupeq


def cheatsheet():
    return 'gblpq: Cholesky re-parameterization that makes GBLUP an ordinary mixed model.'
