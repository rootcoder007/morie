# morie.fn -- function file (rootcoder007/morie)
"""Penalized Poisson log-linear regression."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['poispen', 'poisson_penalized_regression']


def poispen(X, y, lam=1.0, penalty='ridge', n_iter=100, add_intercept=True):
    """Penalized Poisson log-linear regression.

    Formula: l_p = sum_i y_i eta_i - sum_i exp(eta_i) - sum_i log(y_i!) - (lambda/2) sum_j beta_j^2,  eta_i = beta_0 + x_i'beta

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix without an intercept column.
    y : array-like of int
        Observed counts.
    lam : float
        Penalty strength lambda.
    penalty : str
        'ridge' or 'lasso'.
    n_iter : int
        Fixed number of iteratively reweighted least squares iterations.
    add_intercept : bool
        Prepend a column of ones and leave its coefficient unpenalized.

    Returns
    -------
    RichResult
        ``beta``, ``fitted``, ``loglik``, ``n``, ``p``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 7, Sect. 7.5 p. 232.  The book fits this by the second-order approximation of the log-likelihood solved as a weighted least squares problem; the intercept is unpenalized.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.  A FIXED iteration count is used rather than a tolerance stop, so both language arms perform identically many updates.
    """
    out = G.penalized_poisson_fit(X, [float(v) for v in y], lam=float(lam),
                                  penalty=penalty, n_iter=int(n_iter), tol=0.0,
                                  add_intercept=bool(add_intercept))
    Xm = C.mat(X)
    return RichResult(payload={
        "beta": out["beta"], "fitted": out["fitted"], "loglik": out["loglik"],
        "n": len(Xm), "p": len(Xm[0]),
        "method": "Penalized Poisson regression, MVSML Sect. 7.5"})


poisson_penalized_regression = poispen


def cheatsheet():
    return 'poipr: Penalized Poisson log-linear regression.'
