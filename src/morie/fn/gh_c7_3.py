# morie.fn -- function file (rootcoder007/morie)
"""Exponential-link density prior KL support.

Implements sec. 7.1.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_exp_dens_kl"]


def ghosal_exp_dens_kl(coefs0=(0.5, -0.3), coefs=(0.45, -0.25),
                       n_int=800):
    """Densities p_psi propto exp(psi) with psi in a Sobolev-type
    ball: closeness of psi in sup-norm forces small KL(p_psi0; p_psi)
    (sec. 7.1.3). Computes both sup-distance and exact KL for cosine
    expansions on [0,1]. Keys: estimate."""
    def psi(x, cs):
        return sum(c * math.cos((k + 1) * math.pi * x)
                   for k, c in enumerate(cs))
    def dens(cs):
        Z = sum(math.exp(psi((i + 0.5) / n_int, cs))
                for i in range(n_int)) / n_int
        return lambda x: math.exp(psi(x, cs)) / Z
    p0 = dens(list(coefs0))
    p1 = dens(list(coefs))
    kl = sum(p0((i + 0.5) / n_int)
             * math.log(p0((i + 0.5) / n_int)
                        / p1((i + 0.5) / n_int))
             for i in range(n_int)) / n_int
    sup = max(abs(psi((i + 0.5) / 200, list(coefs0))
                  - psi((i + 0.5) / 200, list(coefs)))
              for i in range(200))
    res = RichResult(payload={"estimate": max(kl, 0.0),
                              "sup_norm_gap": sup,
                              "kl_small_when_sup_small":
                                  kl <= 2.0 * sup * math.exp(sup),
                              "method": "exp-link KL support (GvdV 2017 sec. 7.1.3)"})
    return with_describe_pointer(res, "gh_c7_3")


def cheatsheet():
    return "gh_c7_3: Exponential-link density prior KL support"
