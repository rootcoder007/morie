"""Geographically weighted regression."""

from ._richresult import RichResult
from .gwrbas import gwr_basic


def geographically_weighted_regression(y, X, coords, bandwidth, kernel="bisquare", adaptive=False):
    """Geographically weighted regression (:func:`~morie.fn.gwrbas.gwr_basic`).

    ``estimate`` is the matrix of local coefficients; the diagnostics
    (AICc, enp, ...) and standard errors come with it (= ``GWmodel::gwr.basic``).
    """
    r = gwr_basic(y, X, coords, bandwidth, kernel=kernel, adaptive=adaptive)
    return RichResult(payload={"estimate": r["betas"], **dict(r)})


gwrmod = geographically_weighted_regression


def cheatsheet():
    return "gwrmod: geographically weighted regression (GWmodel::gwr.basic)."
