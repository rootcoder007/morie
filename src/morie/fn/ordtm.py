# morie.fn -- function file (rootcoder007/morie)
"""Category probabilities of the ordinal threshold model."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['ordprobs', 'ordinal_threshold_model']


def ordprobs(eta, thresholds, link='probit'):
    """Category probabilities of the ordinal threshold model.

    Formula: p_ic = F(gamma_c + eta_i) - F(gamma_{c-1} + eta_i),  gamma_0 = -inf, gamma_C = +inf

    Parameters
    ----------
    eta : array-like
        Linear predictor x_i'beta for each record.
    thresholds : array-like
        The C - 1 finite thresholds gamma_1 < ... < gamma_{C-1}.
    link : str
        'probit' for the standard normal CDF or 'logistic' for the standard logistic CDF.

    Returns
    -------
    RichResult
        ``probabilities``, ``n``, ``C``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 7, Eq. (7.1) p. 210, which gives both the ordinal probit and the ordinal logistic link on the same page.  Delegates to the chapter routine in morie.fn._gp_core, which was verified against this book in the earlier tranches of this shelf recorded in ledger/SHELF_LEDGER.txt; the page and equation number above are that routine's own, re-read against the chapter PDF here.
    """
    P = G.ordinal_probabilities(eta, thresholds, link=link)
    return RichResult(payload={
        "probabilities": P, "n": len(P), "C": len(P[0]),
        "method": "Ordinal threshold model probabilities, MVSML Eq. (7.1)"})


ordinal_threshold_model = ordprobs


def cheatsheet():
    return 'ordtm: Category probabilities of the ordinal threshold model.'
