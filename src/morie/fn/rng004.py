# morie.fn -- function file (rootcoder007/morie)
"""Skewness of a random process (Rangayyan eq. 3.4)."""


from math import inf, sqrt

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["pdfskew", "rangayyan_ch3_skewness"]


def pdfskew(pdf=None, x=None, lower=-inf, upper=inf):
    """Normalized third central moment of a PDF.

    Rangayyan (2024) eq. (3.4):
        S = (1/sigma^3) integral (eta - mu)^3 p(eta) d eta.

    Symmetric densities give zero; the book reads a negative value as a
    tail to the left of the mode and a positive value as a tail to the
    right.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var)
    if sd <= 0:
        raise ValueError("skewness is undefined for a degenerate density")
    m3 = pdfint(lambda v: (v - mu) ** 3, pdf, x, lower, upper)
    out = {"skewness": float(m3 / sd ** 3), "m3": float(m3), "sd": float(sd),
           "mean": float(mu), "method": "Rangayyan (2024) eq. (3.4)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_skewness = pdfskew  # pre-policy spelling


def cheatsheet():
    return "rng004: skewness of a PDF, Rangayyan eq. (3.4)"
