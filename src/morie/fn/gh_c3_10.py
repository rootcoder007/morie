# morie.fn -- function file (rootcoder007/morie)
"""Normalized completely random measure.

Implements sec. 3.4.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_norm_crm"]


def ghosal_norm_crm(x, n_jumps=400, seed=42):
    """G(A) = M(A)/M(X) for M a completely random measure (GvdV 2017
    sec. 3.4.6). M realized by its jumps: an (approximate) gamma-CRM
    via many small independent gamma jumps at uniform locations --
    normalization gives a Dirichlet-type measure."""
    rng = np.random.default_rng(seed)
    locs = [float(v) for v in rng.uniform(0, 1, n_jumps)._flat()]
    jumps = [float(rng.gamma(1.0 / n_jumps * 4.0, 1.0))
             for _ in range(n_jumps)]
    tot = sum(jumps)
    w = [j / tot for j in jumps]
    half = sum(wi for wi, t in zip(w, locs) if t < 0.5)
    res = RichResult(payload={"estimate": half, "total_mass": 1.0,
                              "n_jumps": n_jumps,
                              "method": "normalized CRM by jump representation (GvdV 2017 sec. 3.4.6)"})
    return with_describe_pointer(res, "gh_c3_10")


def cheatsheet():
    return "gh_c3_10: Normalized completely random measure"


# compact alias per ledger/NAMING.md
ghosalnormcrm = ghosal_norm_crm
