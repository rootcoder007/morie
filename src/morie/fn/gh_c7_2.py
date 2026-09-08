# morie.fn -- function file (rootcoder007/morie)
"""Kernel mixture KL approximation.

Implements sec. 7.1.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kern_mix_kl"]


def ghosal_kern_mix_kl(sd=0.3, grids=(3, 9, 27), span=3.0):
    """KL(p0; f_G) is controlled by how well the mixing measure G
    approximates the true one (sec. 7.1.2). Demo: p0 = N(0, 1+sd^2)
    is an exact normal-location mixture of N(theta, sd^2) over
    G = N(0,1); discretizing G on finer grids drives KL(p0, f_{G_n})
    to zero. Keys: estimate."""
    def npdf(x, m, s):
        z = (x - m) / s
        return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))
    s0 = math.sqrt(1.0 + sd * sd)
    kls = []
    for g in grids:
        pts = [-span + 2.0 * span * (j + 0.5) / g for j in range(g)]
        w = [npdf(t, 0.0, 1.0) for t in pts]
        tot = sum(w)
        w = [v / tot for v in w]
        kl = 0.0
        n_int = 600
        for i in range(n_int):
            x = -6.0 + 12.0 * (i + 0.5) / n_int
            p0x = npdf(x, 0.0, s0)
            fgx = sum(wi * npdf(x, t, sd) for wi, t in zip(w, pts))
            kl += p0x * math.log(p0x / max(fgx, 1e-300)) \
                * 12.0 / n_int
        kls.append(max(kl, 0.0))
    res = RichResult(payload={"estimate": kls[-1],
                              "kl_by_grid": kls,
                              "improving": kls[-1] <= kls[0] + 1e-12,
                              "method": "kernel-mixture KL control (GvdV 2017 sec. 7.1.2)"})
    return with_describe_pointer(res, "gh_c7_2")


def cheatsheet():
    return "gh_c7_2: Kernel mixture KL approximation"
