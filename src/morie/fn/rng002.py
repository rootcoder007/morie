# morie.fn -- function file (rootcoder007/morie)
"""Mean-squared value of a random process (Rangayyan eq. 3.2)."""


from math import inf

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["pdfms", "rangayyan_ch3_mean_squared_value"]


def pdfms(pdf=None, x=None, lower=-inf, upper=inf):
    """Second-order (not central) moment of a PDF.

    Rangayyan (2024) eq. (3.2):  E[eta^2] = integral eta^2 p(eta) d eta.

    The book notes immediately after eq. (3.3) that sigma^2 = E[eta^2] -
    mu^2, so this is the variance only when the mean is zero; both are
    returned so the caller never has to assume which case they are in.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    ms = pdfint(lambda v: v * v, pdf, x, lower, upper)
    out = {"ms": float(ms), "mean": float(mu),
           "variance_from_identity": float(ms - mu * mu),
           "method": "Rangayyan (2024) eq. (3.2)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_mean_squared_value = pdfms  # pre-policy spelling


def cheatsheet():
    return "rng002: mean-squared value of a PDF, Rangayyan eq. (3.2)"
