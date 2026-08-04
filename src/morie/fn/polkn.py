# morie.fn -- function file (rootcoder007/morie)
"""Polynomial kernel matrix."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['polykern', 'polynomial_kernel']


def polykern(X, degree=2, gamma=None, coef0=1.0, Z=None):
    """Polynomial kernel matrix.

    Formula: K(x_i, x_j) = (gamma * x_i'x_j + a)^d

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    degree : int
        Polynomial degree d.
    gamma : float or None
        Scale on the inner product; None uses 1/p.
    coef0 : float
        Constant a added before raising to the power d.
    Z : array-like or None
        Second set of records; None gives the square Gram matrix of X.

    Returns
    -------
    RichResult
        ``K``, ``degree``, ``gamma``, ``coef0``, ``n``, ``m``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Sect. 8.2.2 pp. 255-256 and the worked degree-2 example on p. 256, where the polynomial kernel of degree d with constant a is (gamma x_i'x_j + a)^d and the feature-space dimension is discussed on p. 261.  Read from the chapter PDF, not recalled.
    """
    K = G.kernel_matrix(X, kernel="polynomial", gamma=gamma,
                        degree=int(degree), coef0=float(coef0), Z=Z)
    p = len(G._mat(X)[0])
    g = (1.0 / p) if gamma is None else float(gamma)
    return RichResult(payload={
        "K": K, "degree": int(degree), "gamma": g, "coef0": float(coef0),
        "n": len(K), "m": len(K[0]),
        "method": "Polynomial kernel, MVSML Sect. 8.2.2"})


polynomial_kernel = polykern


def cheatsheet():
    return 'polkn: Polynomial kernel matrix.'
