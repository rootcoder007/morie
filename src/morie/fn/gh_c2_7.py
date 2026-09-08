# morie.fn -- function file (rootcoder007/morie)
"""Bernstein–Feller approximation of a CDF.

Implements sec. 2.3.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_bernstein_feller"]


def ghosal_bernstein_feller(x, F=None, K=30):
    """F_K(x) = sum_k F(k/K) C(K,k) x^k (1-x)^(K-k) (GvdV 2017
    sec. 2.3.4): the Bernstein operator applied to a CDF converges
    uniformly to F on [0,1]."""
    xs = _bnp._flat(x)
    if F is None:
        F = lambda t: t * t          # a genuine CDF on [0,1]
    vals = [_bnp.bernstein_feller_cdf(F, min(max(v, 0.0), 1.0), int(K))
            for v in xs]
    err = max(abs(v - F(min(max(u, 0.0), 1.0)))
              for v, u in zip(vals, xs))
    res = RichResult(payload={"estimate": vals[len(vals) // 2],
                              "F_K": vals, "sup_error": err,
                              "method": "Bernstein-Feller CDF approximation (GvdV 2017 sec. 2.3.4)"})
    return with_describe_pointer(res, "gh_c2_7")


def cheatsheet():
    return "gh_c2_7: Bernstein–Feller approximation of a CDF"
