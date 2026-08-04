# morie.fn -- function file (rootcoder007/morie)
"""Differential entropy of a continuous PDF (Rangayyan eq. 3.6)."""


from math import inf, log

from ._rgcore import checkpdf, pdfint
from ._richresult import RichResult

__all__ = ["diffent", "rangayyan_ch3_entropy_continuous"]


def diffent(pdf=None, x=None, lower=-inf, upper=inf):
    """Differential entropy in bits.

    Rangayyan (2024) eq. (3.6):
        H = - integral p(eta) log2[p(eta)] d eta.

    p log2 p -> 0 as p -> 0, so zero-density points contribute nothing
    rather than raising on log(0).  Unlike the discrete Shannon entropy
    of eq. (3.11) this may be negative -- it is a density, not a
    probability, inside the logarithm.
    """
    ln2 = log(2.0)

    def term(p):
        return 0.0 if p <= 0.0 else -p * log(p) / ln2

    if x is not None:
        from ._rgcore import aslist, gridint
        xs = aslist(x)
        ps = [float(pdf(v)) for v in xs] if callable(pdf) else aslist(pdf)
        h = gridint([term(p) for p in ps], xs)
        mass = gridint(ps, xs)
    else:
        h = pdfint(lambda v: 1.0, lambda v: term(float(pdf(v))),
                   None, lower, upper)
        mass = pdfint(lambda v: 1.0, pdf, None, lower, upper)
    out = {"entropy": float(h), "units": "bits",
           "method": "Rangayyan (2024) eq. (3.6)"}
    out.update(checkpdf(mass))
    return RichResult(payload=out)


rangayyan_ch3_entropy_continuous = diffent  # pre-policy spelling


def cheatsheet():
    return "rng006: differential entropy of a PDF, Rangayyan eq. (3.6)"
