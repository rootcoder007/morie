# morie.fn -- function file (rootcoder007/morie)
"""Bayes's formula for the posterior mass of a set.

Implements sec. 1.3, eq. (1.1) form of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch1_bayes_formula"]


def ghosal_ch1_bayes_formula(B, X, p_theta, Pi):
    """Pi(B | X) = int_B p_theta(X) dPi / int p_theta(X) dPi
    (GvdV 2017 sec. 1.3). ``Pi`` is a discrete prior given as
    (support, weights); ``B`` an indicator over the support."""
    supp, wts = Pi
    supp = _bnp._flat(supp)
    wts = _bnp._flat(wts)
    liks = [p_theta(t, X) for t in supp]
    num = sum(l * w for l, w, t in zip(liks, wts, supp)
              if B(t))
    den = sum(l * w for l, w in zip(liks, wts))
    if den <= 0:
        raise ValueError("zero marginal likelihood on this prior")
    res = RichResult(payload={"posterior": num / den,
                              "marginal": den,
                              "method": "Bayes formula for set mass (GvdV 2017 sec. 1.3)"})
    return with_describe_pointer(res, "ghs001")


def cheatsheet():
    return "ghs001: Bayes's formula for the posterior mass of a set"
