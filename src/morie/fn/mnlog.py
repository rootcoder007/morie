# morie.fn -- function file (rootcoder007/morie)
"""Penalized log-likelihood of the multinomial logistic model."""

from . import _tail1core as C
from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['mnpenlik', 'multinomial_logistic_penalized']


def mnpenlik(X, y, beta0, beta, lam, penalty='ridge'):
    """Penalized log-likelihood of the multinomial logistic model.

    Formula: l_p = l(beta; y) - lambda sum_c beta_c'beta_c (ridge)  or  - lambda sum_c sum_j |beta_cj| (lasso)

    Parameters
    ----------
    X : array-like, shape (n, p)
        Design matrix without an intercept column.
    y : array-like of int
        Observed category index per record, 1-based.
    beta0 : array-like
        Category intercepts.
    beta : array-like, shape (C-1, p) or (C, p)
        Category slope coefficients.
    lam : float
        Penalty strength lambda.
    penalty : str
        'ridge' or 'lasso'.

    Returns
    -------
    RichResult
        ``loglik``, ``penalty``, ``penalized_loglik``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 7, Eq. (7.7) p. 226 for the ridge penalty and Eq. (7.10) p. 227 for the lasso penalty.  The book states on p. 226 that only the slopes are penalized, never the intercepts.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    out = G.penalized_multinomial_loglik(X, [int(v) for v in y], beta0, beta,
                                         float(lam), penalty=penalty)
    return RichResult(payload={
        "loglik": out["loglik"], "penalty": out["penalty"],
        "penalized_loglik": out.get("penalized_loglik",
                                    out["loglik"] - out["penalty"]),
        "n": len(C.mat(X)),
        "method": "Penalized multinomial log-likelihood, MVSML Eq. (7.7)/(7.10)"})


multinomial_logistic_penalized = mnpenlik


def cheatsheet():
    return 'mnlog: Penalized log-likelihood of the multinomial logistic model.'
