# morie.fn -- function file (rootcoder007/morie)
"""Exponential-link density prior.

Implements sec. 2.3.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_exp_link"]


def ghosal_exp_link(x, psi=None):
    """p(x) = exp(psi(x)) / int exp(psi) (GvdV 2017 sec. 2.3.1): a
    positive, normalized density from any bounded function psi. Grid
    normalization; the payload density integrates to 1."""
    import math
    xs = sorted(_bnp._flat(x))
    if psi is None:
        psi = lambda t: math.sin(3.0 * t)
    e = [math.exp(psi(v)) for v in xs]
    Z = sum(0.5 * (e[i] + e[i - 1]) * (xs[i] - xs[i - 1])
            for i in range(1, len(xs)))
    dens = [v / Z for v in e]
    res = RichResult(payload={"estimate": dens[len(dens) // 2],
                              "density": dens, "normalizer": Z,
                              "method": "exponential link density (GvdV 2017 sec. 2.3.1)"})
    return with_describe_pointer(res, "gh_c2_4")


def cheatsheet():
    return "gh_c2_4: Exponential-link density prior"


# compact alias per ledger/NAMING.md
ghosalexplink = ghosal_exp_link
