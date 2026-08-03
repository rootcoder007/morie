# morie.fn -- function file (rootcoder007/morie)
"""Pólya urn view of the Pólya tree.

Implements sec. 3.7.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_polya_urn_pt"]


def ghosal_polya_urn_pt(x, alpha_B=1.0, alpha_total=2.0):
    """Predictive rule P(X_{n+1} in B | X_1..X_n) =
    (alpha_B + n_B) / (alpha + n) (GvdV 2017 sec. 3.7.1): each
    observation adds one unit of mass to its branch, the urn scheme."""
    xs = _bnp._flat(x)
    n = len(xs)
    n_B = sum(1 for v in xs if v < 0.5)     # B = left half
    pred = (alpha_B + n_B) / (alpha_total + n)
    res = RichResult(payload={"estimate": pred, "n_B": n_B, "n": n,
                              "method": "Polya urn predictive (GvdV 2017 sec. 3.7.1)"})
    return with_describe_pointer(res, "gh_c3_13")


def cheatsheet():
    return "gh_c3_13: Pólya urn view of the Pólya tree"
