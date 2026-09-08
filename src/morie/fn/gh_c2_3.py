# morie.fn -- function file (rootcoder007/morie)
"""Increasing-process prior via exponentiated GP.

Implements sec. 2.2.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_increasing_prior"]


def ghosal_gp_increasing_prior(x, length=0.5, seed=42):
    """F(t) = int_0^t exp(W(s)) ds with W a GP path (GvdV 2017
    sec. 2.2.2): strictly increasing by construction. Trapezoid
    integration of the exponentiated draw."""
    import math
    xs = sorted(_bnp._flat(x))
    from .gh_c2_2 import ghosal_gp_prior_def
    w = ghosal_gp_prior_def(xs, length=length, seed=seed)["f"]
    e = [math.exp(v) for v in w]
    F = [0.0]
    for i in range(1, len(xs)):
        F.append(F[-1] + 0.5 * (e[i] + e[i - 1])
                 * (xs[i] - xs[i - 1]))
    res = RichResult(payload={"estimate": F[-1], "F": F,
                              "increasing": all(F[i] <= F[i + 1]
                                                for i in
                                                range(len(F) - 1)),
                              "method": "exponentiated-GP increasing process (GvdV 2017 sec. 2.2.2)"})
    return with_describe_pointer(res, "gh_c2_3")


def cheatsheet():
    return "gh_c2_3: Increasing-process prior via exponentiated GP"
