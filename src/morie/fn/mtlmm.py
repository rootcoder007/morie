# morie.fn -- function file (rootcoder007/morie)
"""Multi-trait linear mixed model."""

from . import _tail1core as C
from . import _gp_core as GC

from ._richresult import RichResult

__all__ = ['mtlmmfit', 'multi_trait_lmm', 'multitraitlmm']


def mtlmmfit(Y, Z, G, Sigma_T, R_T, X=None):
    """Multi-trait linear mixed model.

    Formula: Y = (1 (x) I_T) mu + X beta + Z b + eps,  b ~ N(0, G (x) Sigma_T),  eps ~ N(0, I_J (x) R_T)

    Parameters
    ----------
    Y : array-like, shape (J, T)
        Lines by traits.
    Z : array-like, shape (J, q)
        Design matrix of lines.
    G : array-like, shape (q, q)
        Genomic relationship matrix.
    Sigma_T : array-like, shape (T, T)
        Genetic covariance between traits.
    R_T : array-like, shape (T, T)
        Residual covariance between traits.
    X : array-like or None
        Extra fixed-effect columns; None uses only the trait intercepts.

    Returns
    -------
    RichResult
        ``mu``, ``beta``, ``b``, ``J``, ``T``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 5, Eq. (5.5)/(5.5a) p. 153: the traits of each line are stacked, giving a mixed model with Kronecker-structured covariances.  The book notes on p. 153 that when Sigma_T and R_T are diagonal this is equivalent to fitting each trait separately.  The solution is in the stacked ordering (line 1 traits, line 2 traits, ...).  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    out = GC.multitrait_model(Y, Z, G, Sigma_T, R_T, X=X)
    Ym = C.mat(Y)
    return RichResult(payload={
        "mu": out["mu"], "beta": out["beta"], "b": out["b"],
        "J": len(Ym), "T": len(Ym[0]),
        "method": "Multi-trait linear mixed model, MVSML Eq. (5.5)"})


multi_trait_lmm = mtlmmfit
multitraitlmm = mtlmmfit


def cheatsheet():
    return 'mtlmm: Multi-trait linear mixed model.'
