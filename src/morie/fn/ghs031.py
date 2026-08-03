# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree mixture posterior density.

Implements sec. 3.7.2 (mixtures; eq. 3.23 applied to G_theta-transformed data) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_ch3_polya_tree_mixture_post_density"]


def ghosal_ch3_polya_tree_mixture_post_density(x, data, g_theta,
                                               G_theta, a_of_level=None,
                                               depth=8):
    """E(p(x) | theta, X) = g_theta(x) * prod_j (2 a_j + 2 N*_j) /
    (2 a_j + N*_{j-1}) where N* counts the transformed data
    U_i = G_theta(X_i) sharing G_theta(x)'s dyadic path -- a PT on the
    G_theta scale pushed back through the parametric family
    (GvdV 2017 sec. 3.7.2). Keys: posterior."""
    if a_of_level is None:
        a_of_level = lambda m: float(m * m)
    x0 = _bnp._flat(x)[0]
    d = _bnp._flat(data)
    u0 = float(G_theta(x0))
    us = [float(G_theta(v)) for v in d]
    n = len(us)
    counts = _bnp.pt_path_counts(u0, us, int(depth))
    core = _bnp.pt_density_posterior(u0, a_of_level, counts, n,
                                     int(depth))
    dens = float(g_theta(x0)) * core
    res = RichResult(payload={"estimate": dens, "posterior": dens,
                              "uniform_scale_density": core,
                              "method": "PT mixture posterior density (GvdV 2017 sec. 3.7.2)"})
    return with_describe_pointer(res, "ghs031")


def cheatsheet():
    return "ghs031: Pólya tree mixture posterior density"
