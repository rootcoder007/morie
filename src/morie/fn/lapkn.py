# morie.fn -- function file (rootcoder007/morie)
"""Exponential (Laplace) kernel matrix."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['expkern', 'laplacian_kernel']


def expkern(X, gamma=None, Z=None):
    """Exponential (Laplace) kernel matrix.

    Formula: K(x_i, x_j) = exp(-gamma * ||x_i - x_j||)

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    gamma : float or None
        Bandwidth; None uses 1/p.
    Z : array-like or None
        Second set of records; None gives the square Gram matrix of X.

    Returns
    -------
    RichResult
        ``K``, ``gamma``, ``n``, ``m``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Sect. 8.2.2 p. 264: the book calls this the Exponential Kernel, K(x_i, x_j) = exp(-gamma ||x_i - x_j||), and notes it is close to the Gaussian kernel.  NOTE: the placeholder this replaced was named for the Laplacian kernel; the book's display uses the Euclidean norm, not the L1 norm that the name Laplacian kernel usually implies, and the book's form is what is implemented.  Read from the chapter PDF, not recalled.
    """
    K = G.kernel_matrix(X, kernel="exponential", gamma=gamma, Z=Z)
    p = len(G._mat(X)[0])
    g = (1.0 / p) if gamma is None else float(gamma)
    return RichResult(payload={
        "K": K, "gamma": g, "n": len(K), "m": len(K[0]),
        "method": "Exponential (Laplace) kernel, MVSML Sect. 8.2.2"})


laplacian_kernel = expkern


def cheatsheet():
    return 'lapkn: Exponential (Laplace) kernel matrix.'
