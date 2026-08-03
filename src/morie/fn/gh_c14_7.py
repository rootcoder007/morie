# morie.fn -- function file (rootcoder007/morie)
"""Species-sampling mixture.

Implements sec. 14.2.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ssp_mix"]


def ghosal_ssp_mix(x, weights, atoms, kernel_sd=0.3):
    """f(x) = int K(x; theta) dG(theta) with G a species-sampling
    process (sec. 14.2.2): normal-kernel mixture over the atoms.
    Keys: estimate."""
    p = _bnp.normalize_weights(weights)
    th = _bnp._flat(atoms)
    xs = _bnp._flat(x)
    def npdf(v, m):
        z = (v - m) / kernel_sd
        return math.exp(-0.5 * z * z) / (kernel_sd
                                         * math.sqrt(2 * math.pi))
    dens = [sum(pi * npdf(v, t) for pi, t in zip(p, th))
            for v in xs]
    res = RichResult(payload={"estimate": dens[0],
                              "density": dens,
                              "method": "SSP mixture density (GvdV 2017 sec. 14.2.2)"})
    return with_describe_pointer(res, "gh_c14_7")


def cheatsheet():
    return "gh_c14_7: Species-sampling mixture"
