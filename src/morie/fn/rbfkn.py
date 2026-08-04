# morie.fn -- function file (rootcoder007/morie)
"""Gaussian (radial basis) kernel matrix."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['rbfkern', 'rbf_kernel']


def rbfkern(X, gamma=None, Z=None):
    """Gaussian (radial basis) kernel matrix.

    Formula: K(x_i, x_j) = exp(-gamma * ||x_i - x_j||^2)

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    gamma : float or None
        Bandwidth; None uses 1/p, p the number of columns of X.
    Z : array-like or None
        Second set of records; None gives the square Gram matrix of X.

    Returns
    -------
    RichResult
        ``K``, ``gamma``, ``n``, ``m``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Sect. 8.2.2 pp. 263-264.  The book prints the Gaussian kernel through its R implementation K.radial, which computes exp(-gamma * ||x1 - x2||^2); gamma, not a variance-style bandwidth, is the parameter the text discusses on p. 264.  The placeholder this replaced carried a 1/(2h^2) parameterization that the book does not use.  Read from the chapter PDF, not recalled.
    """
    K = G.kernel_matrix(X, kernel="gaussian", gamma=gamma, Z=Z)
    p = len(G._mat(X)[0])
    g = (1.0 / p) if gamma is None else float(gamma)
    return RichResult(payload={
        "K": K, "gamma": g, "n": len(K), "m": len(K[0]),
        "method": "Gaussian (RBF) kernel, MVSML Sect. 8.2.2"})


rbf_kernel = rbfkern


def cheatsheet():
    return 'rbfkn: Gaussian (radial basis) kernel matrix.'
