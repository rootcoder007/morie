# morie.fn -- function file (rootcoder007/morie)
"""Bayesian kernel BLUP: the closed-form conditional mean of the kernel effects."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['rkhsbayes', 'rkhs_bayesian_kernel']


def rkhsbayes(y, K, sigma2_u=1.0, sigma2_e=1.0, mu=None):
    """Bayesian kernel BLUP: the closed-form conditional mean of the kernel effects.

    Formula: Ktilde = (K^-1/sigma2_u + I/sigma2_e)^-1;  utilde = Ktilde (y - 1 mu)/sigma2_e

    Parameters
    ----------
    y : array-like
        Response vector of length n.
    K : array-like, shape (n, n)
        Kernel matrix.
    sigma2_u : float
        Variance component of the kernel effects.
    sigma2_e : float
        Residual variance component.
    mu : float or None
        Intercept; None uses the mean of y.

    Returns
    -------
    RichResult
        ``u``, ``K_tilde``, ``mu``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8, Eq. (8.8) p. 281 and its full conditionals on p. 282: y = 1 mu + u + e with u ~ N(0, sigma2_u K), which is kernel ridge regression with lambda = sigma2_e/sigma2_u.  Only the CLOSED-FORM conditional mean is computed here, not the Gibbs sampler: a sampler would make the two language arms depend on matching random number streams, and the conditional mean is the quantity the equation defines.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    ut, Kt = G.bayesian_kernel_blup(y, K, sigma2_u=float(sigma2_u),
                                    sigma2_e=float(sigma2_e), gibbs=False)
    yv = C.vec(y)
    m = (sum(yv) / len(yv)) if mu is None else float(mu)
    return RichResult(payload={
        "u": ut, "K_tilde": Kt, "mu": m, "n": len(yv),
        "method": "Bayesian kernel BLUP conditional mean, MVSML Eq. (8.8)"})


rkhs_bayesian_kernel = rkhsbayes


def cheatsheet():
    return 'rkhsb: Bayesian kernel BLUP: the closed-form conditional mean of the kernel effects.'
