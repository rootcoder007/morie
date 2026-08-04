# morie.fn -- function file (rootcoder007/morie)
"""Default hyperparameters of the Bayesian ridge regression prior."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['brrhyper', 'brr_prior_posterior']


def brrhyper(y, R2=0.5, nu=5.0, nu_beta=5.0):
    """Default hyperparameters of the Bayesian ridge regression prior.

    Formula: S = Var(Y)(1 - R2)(nu + 2);  S_beta = Var(Y) R2 (nu_beta + 2)

    Parameters
    ----------
    y : array-like
        Response vector of length n.
    R2 : float
        Prior proportion of the phenotypic variance explained by the markers.
    nu : float
        Degrees of freedom of the scaled inverse chi-square prior on the residual variance.
    nu_beta : float
        Degrees of freedom of the prior on the marker-effect variance.

    Returns
    -------
    RichResult
        ``S``, ``S_beta``, ``nu``, ``nu_beta``, ``var_y``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 6, the BGLR default rules quoted on pp. 175 and 184: the scale of the residual prior is Var(Y)(1 - R2)(nu + 2) and, for the BRR, the scale of the marker-effect prior is Var(Y) R2 (nu_beta + 2).  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    out = G.brr_hyperparameters(y, R2=float(R2), nu=float(nu),
                                nu_beta=float(nu_beta))
    yv = C.vec(y)
    return RichResult(payload={
        "S": out["S"], "S_beta": out["S_beta"], "nu": out["nu"],
        "nu_beta": out["nu_beta"], "var_y": C.var(yv, ddof=1), "n": len(yv),
        "method": "BRR prior hyperparameters, MVSML Chap. 6 pp. 175, 184"})


brr_prior_posterior = brrhyper


def cheatsheet():
    return 'brrpf: Default hyperparameters of the Bayesian ridge regression prior.'
