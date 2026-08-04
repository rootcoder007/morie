# morie.fn -- function file (rootcoder007/morie)
"""Principal component compression of a marker matrix."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['pcadim', 'pca_dimensionality_reduction']


def pcadim(X, k=None):
    """Principal component compression of a marker matrix.

    Formula: Q = X'X/(n-1) on scaled columns; W the eigenvectors of Q; PC = X W; keep the first k columns

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    k : int or None
        Number of components retained; None keeps all.

    Returns
    -------
    RichResult
        ``scores``, ``loadings``, ``eigenvalues``, ``prop_var``, ``cum_prop``, ``k``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Sect. 2.8 pp. 63-64.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.  Eigenvectors are sign-fixed so the two language arms agree; a repeated eigenvalue leaves its loadings determined only up to a rotation within the eigenspace and is not a stable quantity in either language.
    """
    out = G.pca_compress(X, k=k)
    Xm = C.mat(X)
    return RichResult(payload={
        "scores": out.get("scores", out.get("PC")),
        "loadings": out["loadings"] if "loadings" in out else out.get("W"),
        "eigenvalues": out["eigenvalues"] if "eigenvalues" in out else out.get("values"),
        "prop_var": out.get("prop_var"), "cum_prop": out.get("cum_prop"),
        "k": out.get("k"), "n": len(Xm), "p": len(Xm[0]),
        "method": "PCA compression, MVSML Sect. 2.8"})


pca_dimensionality_reduction = pcadim


def cheatsheet():
    return 'pcadr: Principal component compression of a marker matrix.'
