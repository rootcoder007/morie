# morie.fn -- function file (rootcoder007/morie)
"""Kurtosis of a random process (Rangayyan eq. 3.5)."""


from math import inf, sqrt

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["pdfkurt", "rangayyan_ch3_kurtosis"]


def pdfkurt(pdf=None, x=None, lower=-inf, upper=inf):
    """Normalized fourth central moment of a PDF.

    Rangayyan (2024) eq. (3.5):
        K = (1/sigma^4) integral (eta - mu)^4 p(eta) d eta.

    The book states the Gaussian value is 3 and defines the kurtosis
    excess K' = K - 3, positive for a strongly peaked heavy-tailed
    density and negative for a near-uniform one; both are returned.
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var)
    if sd <= 0:
        raise ValueError("kurtosis is undefined for a degenerate density")
    m4 = pdfint(lambda v: (v - mu) ** 4, pdf, x, lower, upper)
    k = float(m4 / sd ** 4)
    out = {"kurtosis": k, "excess": k - 3.0, "m4": float(m4),
           "sd": float(sd), "mean": float(mu),
           "method": "Rangayyan (2024) eq. (3.5)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_kurtosis = pdfkurt  # pre-policy spelling


def cheatsheet():
    return "rng005: kurtosis of a PDF, Rangayyan eq. (3.5)"
