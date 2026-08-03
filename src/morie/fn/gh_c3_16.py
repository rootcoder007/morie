# morie.fn -- function file (rootcoder007/morie)
"""Evenly split Pólya tree.

Implements sec. 3.7.4 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_evsplit_pt"]


def ghosal_evsplit_pt(x, data=None, depth=6, a_scale=1.0):
    """Canonical PT*(alpha, a_m) with alpha_e0 = alpha_e1 = a_m at
    level m (GvdV 2017 sec. 3.7.4): symmetric splits give the uniform
    prior mean density; with data, the posterior mean density follows
    the alpha -> alpha + N updating of sec. 3.7."""
    xs = _bnp._flat(x)
    x0 = xs[0] if xs else 0.3
    if data is None:
        data = []
    n = len(data)
    counts = _bnp.pt_path_counts(x0, data, depth) if n else [0] * depth
    dens = _bnp.pt_density_posterior(x0, lambda m: a_scale * m * m,
                                     counts, n, depth)
    res = RichResult(payload={"estimate": dens, "depth": depth,
                              "n": n,
                              "method": "evenly split PT posterior density (GvdV 2017 sec. 3.7.4)"})
    return with_describe_pointer(res, "gh_c3_16")


def cheatsheet():
    return "gh_c3_16: Evenly split Pólya tree"
