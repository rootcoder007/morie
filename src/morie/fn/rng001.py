# morie.fn -- function file (rootcoder007/morie)
"""Mean of a random process from its PDF (Rangayyan eq. 3.1)."""


from math import inf

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["pdfmean", "rangayyan_ch3_mean_continuous"]


def pdfmean(pdf=None, x=None, lower=-inf, upper=inf):
    """First-order moment of a PDF.

    Rangayyan (2024) eq. (3.1):  mu_eta = E[eta] = integral eta p(eta) d eta.

    Parameters
    ----------
    pdf : callable or array-like
        The density.  Callable is integrated adaptively between ``lower``
        and ``upper``; array-like is read as densities tabulated at ``x``.
    x : array-like, optional
        Abscissae for a tabulated density.
    lower, upper : float
        Limits for the callable form.  Infinite limits are truncated at
        +/- 40, which holds any density whose scale is order unity; pass
        finite limits when the density lives elsewhere.

    Returns
    -------
    RichResult
        ``mean`` plus the integrated mass of the density, which is the
        only cheap check that the input really is a PDF.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    out = {"mean": float(mu), "method": "Rangayyan (2024) eq. (3.1)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_mean_continuous = pdfmean  # pre-policy spelling


def cheatsheet():
    return "rng001: mean of a PDF, Rangayyan eq. (3.1)"
