# morie.fn -- function file (rootcoder007/morie)
"""Restricted (residual) maximum likelihood log-likelihood of a mixed model."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['remlik', 'reml_log_likelihood']


def remlik(X, Z, y, D, R=None):
    """Restricted (residual) maximum likelihood log-likelihood of a mixed model.

    Formula: l_R(theta; y) = -0.5 log|X'V^-1 X| - 0.5 log|V| - 0.5 (y - X betatilde)' V^-1 (y - X betatilde)

    Parameters
    ----------
    X : array-like, shape (n, p)
        Fixed-effect design matrix.
    Z : array-like, shape (n, q)
        Random-effect design matrix.
    y : array-like
        Response vector of length n.
    D : array-like, shape (q, q)
        Variance-covariance matrix of the random effects.
    R : array-like or None
        Residual variance-covariance matrix; None uses the identity.

    Returns
    -------
    RichResult
        ``loglik``, ``beta``, ``n``, ``p``, ``q``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 5, Sect. 5.2.1.2 p. 146.  REML differs from the ML log-likelihood of Eq. (5.2) by the -0.5 log|X'V^-1 X| term, which is what removes the downward bias of the ML variance estimate; betatilde is the generalized least squares estimator.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    ll, beta = G.reml_loglik(X, Z, y, D, R=R)
    Xm = C.mat(X); Zm = C.mat(Z)
    return RichResult(payload={
        "loglik": ll, "beta": beta, "n": len(Xm), "p": len(Xm[0]),
        "q": len(Zm[0]), "method": "REML log-likelihood, MVSML Sect. 5.2.1.2"})


reml_log_likelihood = remlik


def cheatsheet():
    return 'remlf: Restricted (residual) maximum likelihood log-likelihood of a mixed model.'
