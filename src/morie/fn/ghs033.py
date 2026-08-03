# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree mixture of the second kind.

Implements sec. 3.7.2 (theta-indexed prior mean, eq. 3.22 form) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_polya_tree_mixture_second_kind"]


def ghosal_ch3_polya_tree_mixture_second_kind(x, alpha_path_of_theta,
                                              thetas, weights=None):
    """g_theta(x) = prod_j 2 alpha_{x_1..x_j}(theta) /
    (alpha_..0(theta) + alpha_..1(theta)): the eq. (3.22) prior mean
    density with theta-dependent parameters, mixed over theta
    (GvdV 2017 sec. 3.7.2). ``alpha_path_of_theta(theta, x)`` returns
    the (alpha_taken, alpha_other) pairs. Keys: distribution."""
    x0 = _bnp._flat(x)[0]
    ths = list(thetas)
    if weights is None:
        weights = [1.0 / len(ths)] * len(ths)
    w = _bnp.normalize_weights(weights)
    per = []
    for th in ths:
        m1 = 1.0
        for a_take, a_other in alpha_path_of_theta(th, x0):
            m1 *= 2.0 * float(a_take) / (float(a_take)
                                         + float(a_other))
        per.append(m1)
    mix = sum(wi * gi for wi, gi in zip(w, per))
    res = RichResult(payload={"estimate": mix, "distribution": mix,
                              "per_theta": per,
                              "method": "PT mixture second kind (GvdV 2017 sec. 3.7.2)"})
    return with_describe_pointer(res, "ghs033")


def cheatsheet():
    return "ghs033: Pólya tree mixture of the second kind"
