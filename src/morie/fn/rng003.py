# morie.fn -- function file (rootcoder007/morie)
"""Variance of a random process (Rangayyan eq. 3.3)."""


from math import inf, sqrt

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["pdfvar", "rangayyan_ch3_variance_continuous"]


def pdfvar(pdf=None, x=None, lower=-inf, upper=inf):
    """Second central moment of a PDF, and the SD and CV that follow.

    Rangayyan (2024) eq. (3.3):
        sigma^2 = E[(eta - mu)^2] = integral (eta - mu)^2 p(eta) d eta.

    The book defines the coefficient of variation as sigma/mu in the same
    paragraph and warns it diverges as mu -> 0, so ``cv`` is None once the
    mean is negligible against the SD of the process -- a quadrature
    residue of 1e-19 on a symmetric density would otherwise be reported
    as a CV of 1e19 rather than as "undefined here".
    """
    mass = pdfint(lambda v: 1.0, pdf, x, lower, upper)
    mu = pdfint(lambda v: v, pdf, x, lower, upper)
    var = pdfint(lambda v: (v - mu) ** 2, pdf, x, lower, upper)
    sd = sqrt(var) if var > 0 else 0.0
    cv = float(sd / mu) if abs(mu) > 1e-9 * max(sd, 1.0) else None
    out = {"variance": float(var), "sd": float(sd), "mean": float(mu),
           "cv": cv, "method": "Rangayyan (2024) eq. (3.3)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_variance_continuous = pdfvar  # pre-policy spelling


def cheatsheet():
    return "rng003: variance of a PDF, Rangayyan eq. (3.3)"
