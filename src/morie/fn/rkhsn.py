# morie.fn -- function file (rootcoder007/morie)
"""Squared RKHS norm of a kernel expansion."""

import math

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['rkhsnorm', 'rkhs_norm']


def rkhsnorm(beta, K):
    """Squared RKHS norm of a kernel expansion.

    Formula: ||f||_H^2 = sum_i sum_j beta_i beta_j K(x_i, x_j) = beta' K beta

    Parameters
    ----------
    beta : array-like
        Kernel expansion coefficients, length n.
    K : array-like, shape (n, n)
        Gram matrix.

    Returns
    -------
    RichResult
        ``norm2``, ``norm``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Eq. (8.2) p. 254: the squared norm of f in the reproducing kernel Hilbert space is beta'K beta.  Read from the chapter PDF, not recalled.
    """
    n2 = G.rkhs_norm(beta, K)
    if n2 < 0.0:
        raise ValueError("beta'K beta is negative: K is not positive semi-definite")
    return RichResult(payload={
        "norm2": n2, "norm": math.sqrt(n2), "n": len(G._flat(beta)),
        "method": "Squared RKHS norm, MVSML Eq. (8.2)"})


rkhs_norm = rkhsnorm


def cheatsheet():
    return 'rkhsn: Squared RKHS norm of a kernel expansion.'
