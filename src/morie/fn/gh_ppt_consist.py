# morie.fn -- function file (rootcoder007/morie)
"""Pólya tree posterior contraction.

Implements sec. 7.2.3 (canonical PT contraction) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_polya_tree_consist_rate"]


def ghosal_polya_tree_consist_rate(ns=(50, 200, 800), depth=6,
                                   seed=42):
    """Canonical PT*(lambda, m^2) posteriors contract at the truth
    (sec. 7.2.3): the posterior mean density (eq. 3.23) moves toward
    p0 as n grows; L1-type error over a query grid decreases.
    Truth: triangular density 2x on [0,1]. Keys: estimate."""
    rng = np.random.default_rng(seed)
    a_of = lambda m: float(m * m)
    errs = []
    for n in ns:
        data = [math.sqrt(float(rng.uniform(0, 1)))
                for _ in range(n)]           # p0(x) = 2x
        err = 0.0
        for j in range(8):
            x = (j + 0.5) / 8.0
            counts = _bnp.pt_path_counts(x, data, depth)
            dens = _bnp.pt_density_posterior(x, a_of, counts, n,
                                             depth)
            err += abs(dens - 2.0 * x) / 8.0
        errs.append(err)
    res = RichResult(payload={"estimate": errs[-1],
                              "l1_error_by_n": errs,
                              "contracting": errs[-1] < errs[0],
                              "method": "PT posterior contraction (GvdV 2017 sec. 7.2.3)"})
    return with_describe_pointer(res, "gh_ppt_consist")


def cheatsheet():
    return "gh_ppt_consist: Pólya tree posterior contraction"
