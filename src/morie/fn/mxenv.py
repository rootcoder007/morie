# morie.fn -- function file (rootcoder007/morie)
"""Multi-environment genomic model with a genotype-by-environment term."""

from . import _tail1core as C
from . import _gp_core as GC

from ._richresult import RichResult

__all__ = ['gxeblup', 'multi_env_model', 'multienvmodel']


def gxeblup(y, X_E, Z_L, Z_EL, G, sigma2_g, Sigma_E, sigma2_e=1.0):
    """Multi-environment genomic model with a genotype-by-environment term.

    Formula: Y = 1 mu + X_E beta_E + Z_L b_1 + Z_EL b_2 + eps,  b_1 ~ N(0, sigma2_g G),  b_2 ~ N(0, Sigma_E (x) G)

    Parameters
    ----------
    y : array-like
        Response vector of length n.
    X_E : array-like or None
        Environment design matrix; None or empty uses an intercept only.
    Z_L : array-like, shape (n, J)
        Design matrix of lines.
    Z_EL : array-like, shape (n, J*E)
        Design matrix of the line-by-environment interaction.
    G : array-like, shape (J, J)
        Genomic relationship matrix.
    sigma2_g : float
        Genomic variance component.
    Sigma_E : array-like, shape (E, E)
        Genetic covariance between environments.
    sigma2_e : float
        Residual variance component.

    Returns
    -------
    RichResult
        ``beta``, ``b_lines``, ``b_gxe``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 5, Eq. (5.4) p. 150.  The two random terms are stacked into one Z = [Z_L  Z_EL] with a block-diagonal Sigma and solved as Eq. (5.1).  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    out = GC.gxe_blup_model(y, X_E, Z_L, Z_EL, G,
                            float(sigma2_g), Sigma_E, sigma2_e=float(sigma2_e))
    return RichResult(payload={
        "beta": out["beta"], "b_lines": out["b_lines"], "b_gxe": out["b_gxe"],
        "n": len(C.vec(y)),
        "method": "Multi-environment GxE model, MVSML Eq. (5.4)"})


multi_env_model = gxeblup
multienvmodel = gxeblup


def cheatsheet():
    return 'mxenv: Multi-environment genomic model with a genotype-by-environment term.'
