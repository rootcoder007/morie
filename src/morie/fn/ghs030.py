# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree posterior mean density.

Implements eq. (3.23), p.50 (conjugacy Theorem 3.21) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_polya_tree_posterior_density"]


def ghosal_ch3_polya_tree_posterior_density(x, data, a_of_level=None,
                                            depth=8):
    """E(p(x) | X) = prod_j (2 a_j + 2 N_{x_1..x_j}) /
    (2 a_j + N_{x_1..x_{j-1}}) (eq. 3.23) -- the canonical-PT
    posterior via alpha* = alpha + N (Theorem 3.21).
    Keys: posterior."""
    if a_of_level is None:
        a_of_level = lambda m: float(m * m)
    xs = _bnp._flat(x)
    d = _bnp._flat(data)
    n = len(d)
    counts = _bnp.pt_path_counts(xs[0], d, int(depth))
    dens = _bnp.pt_density_posterior(xs[0], a_of_level, counts, n,
                                     int(depth))
    res = RichResult(payload={"estimate": dens, "posterior": dens,
                              "path_counts": counts, "n": n,
                              "method": "PT posterior density (GvdV 2017 eq. 3.23)"})
    return with_describe_pointer(res, "ghs030")


def cheatsheet():
    return "ghs030: Pólya tree posterior mean density"
