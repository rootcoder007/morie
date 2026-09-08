# morie.fn -- function file (rootcoder007/morie)
"""Kernel covariance of a replicated-line predictor."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['kernblup', 'kernel_blup', 'kernelblup']


def kernblup(Z, K, sigma2_u=1.0):
    """Kernel covariance of a replicated-line predictor.

    Formula: K_* = Var(Z u) = sigma2_u * Z K Z'

    Parameters
    ----------
    Z : array-like, shape (n, J)
        Incidence matrix mapping records to lines.
    K : array-like, shape (J, J)
        Kernel (relationship) matrix between lines.
    sigma2_u : float
        Variance component of the line effects.

    Returns
    -------
    RichResult
        ``K_star``, ``n``, ``J``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Eq. (8.9) p. 282: with replicated individuals the model is Y = 1 mu + Z u + e; BGLR cannot take that predictor directly, so the covariance of the predictor, Z K Z', is precomputed and used as the kernel.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    Ks = G.kernel_blup_replicated(Z, K, float(sigma2_u))
    Zm = C.mat(Z)
    return RichResult(payload={
        "K_star": Ks, "n": len(Zm), "J": len(Zm[0]),
        "method": "Replicated-line kernel covariance, MVSML Eq. (8.9)"})


kernel_blup = kernblup
kernelblup = kernblup


def cheatsheet():
    return 'kblup: Kernel covariance of a replicated-line predictor.'
