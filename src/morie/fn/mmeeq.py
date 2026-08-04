# morie.fn -- function file (rootcoder007/morie)
"""Henderson's mixed model equations for the BLUE and the BLUP."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['hendmme', 'henderson_mme_eq2_2']


def hendmme(X, Z, y, Sigma_inv, R_inv=None):
    """Henderson's mixed model equations for the BLUE and the BLUP.

    Formula: [[X'R^-1 X, X'R^-1 Z], [Z'R^-1 X, Z'R^-1 Z + Sigma^-1]] [beta; u] = [X'R^-1 y; Z'R^-1 y]

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix of fixed effects.
    Z : array-like, shape (n, q)
        Design matrix of random effects.
    y : array-like
        Response vector of length n.
    Sigma_inv : array-like, shape (q, q)
        Inverse of the random-effect variance-covariance matrix.
    R_inv : array-like or None
        Inverse residual variance-covariance matrix; None uses the identity.

    Returns
    -------
    RichResult
        ``beta``, ``u``, ``fitted``, ``n``, ``p``, ``q``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Eq. (2.2) p. 37: Henderson's mixed model equations, whose solution for beta is the BLUE and for u is the BLUP.  Delegates to the chapter-2 MME solver already verified against the book for this shelf.  Read from the chapter PDF, not recalled.
    """
    out = G.mme_solve(X, Z, y, Sigma_inv, R_inv=R_inv)
    b = out["beta"] if isinstance(out, dict) else out[0]
    uu = out["u"] if isinstance(out, dict) else out[1]
    Xm = C.mat(X); Zm = C.mat(Z)
    fit = [a + c for a, c in zip(C.matvec(Xm, b), C.matvec(Zm, uu))]
    return RichResult(payload={
        "beta": b, "u": uu, "fitted": fit,
        "n": len(Xm), "p": len(Xm[0]), "q": len(Zm[0]),
        "method": "Henderson mixed model equations, MVSML Eq. (2.2)"})


henderson_mme_eq2_2 = hendmme


def cheatsheet():
    return "mmeeq: Henderson's mixed model equations for the BLUE and the BLUP."
