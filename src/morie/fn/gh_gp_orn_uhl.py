# morie.fn -- function file (rootcoder007/morie)
"""Ornstein-Uhlenbeck process.

Implements Example 11.7 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_ornstein_uhlenbeck"]


def ghosal_gp_ornstein_uhlenbeck(theta=1.0, ts=(0.2, 0.5, 0.9)):
    """OU: K(s, t) = (2 theta)^{-1} e^{-theta |s - t|}; equivalently
    W_t = (2 theta)^{-1/2} e^{-theta t} B(e^{2 theta t}) (Ex 11.7).
    Verifies the BM-representation covariance equals the kernel and
    the Markov screening property K(s,u) = K(s,t) K(t,u)/K(t,t) for
    s < t < u. Keys: estimate."""
    th = float(theta)
    def K(s, t):
        return math.exp(-th * abs(s - t)) / (2.0 * th)
    # representation covariance: for s <= t,
    # Cov = (2th)^{-1} e^{-th(s+t)} min(e^{2th s}, e^{2th t})
    #     = (2th)^{-1} e^{-th(t-s)}
    s, t, u = ts
    rep = math.exp(-th * (t - s)) / (2.0 * th)
    markov_gap = abs(K(s, u) - K(s, t) * K(t, u) / K(t, t))
    res = RichResult(payload={"estimate": K(s, t),
                              "representation_gap":
                                  abs(K(s, t) - rep),
                              "markov_gap": markov_gap,
                              "method": "Ornstein-Uhlenbeck kernel (GvdV 2017 Ex 11.7)"})
    return with_describe_pointer(res, "gh_gp_orn_uhl")


def cheatsheet():
    return "gh_gp_orn_uhl: Ornstein-Uhlenbeck process"
