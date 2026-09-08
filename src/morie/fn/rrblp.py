# morie.fn -- function file (rootcoder007/morie)
"""SNP-BLUP (ridge regression BLUP) marker effects and breeding values."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['snpblup', 'rrblup_marker_effects']


def snpblup(X, y, M, sigma2_m, sigma2_e=1.0):
    """SNP-BLUP (ridge regression BLUP) marker effects and breeding values.

    Formula: Z = M (scaled markers), Sigma = sigma2_M I in Henderson's equations; GEBV = M uhat

    Parameters
    ----------
    X : array-like, shape (n, p)
        Fixed-effect design matrix.
    y : array-like
        Response vector of length n.
    M : array-like, shape (n, m)
        Scaled marker matrix.
    sigma2_m : float
        Marker-effect variance component.
    sigma2_e : float
        Residual variance component.

    Returns
    -------
    RichResult
        ``beta``, ``marker_effects``, ``gebv``, ``n``, ``m``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 2, Eq. (2.4) p. 53: SNP-BLUP replaces Z by the scaled marker matrix M and Sigma by sigma2_M I in Eq. (2.2); the genomic estimated breeding value is M uhat.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    beta, u, gebv = G.snp_blup_gebv(X, y, M, float(sigma2_m), float(sigma2_e))
    Mm = C.mat(M)
    return RichResult(payload={
        "beta": beta, "marker_effects": u, "gebv": gebv,
        "n": len(Mm), "m": len(Mm[0]),
        "method": "SNP-BLUP marker effects, MVSML Eq. (2.4)"})


rrblup_marker_effects = snpblup


def cheatsheet():
    return 'rrblp: SNP-BLUP (ridge regression BLUP) marker effects and breeding values.'
