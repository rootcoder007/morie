# morie.fn -- function file (rootcoder007/morie)
"""Dirichlet vector by gamma normalization.

Implements sec. 3.3.1 (eq. 3.1 route; Proposition G.2) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_dir_simplex"]


def ghosal_dir_simplex(x, alpha=None, seed=42):
    """(p_1..p_k) ~ Dir(alpha): p_j = G_j / sum G_i with independent
    G_j ~ Ga(alpha_j, 1) (GvdV 2017 sec. 3.3.1, Proposition G.2)."""
    if alpha is None:
        alpha = [1.0, 2.0, 3.0]
    a = _bnp._flat(alpha)
    rng = np.random.default_rng(seed)
    g = [float(rng.gamma(ai, 1.0)) for ai in a]
    p = _bnp.normalize_weights(g)
    res = RichResult(payload={"estimate": p[0], "p": p,
                              "alpha": a,
                              "method": "Dirichlet by gamma normalization (GvdV 2017 sec. 3.3.1)"})
    return with_describe_pointer(res, "gh_c3_3")


def cheatsheet():
    return "gh_c3_3: Dirichlet vector by gamma normalization"
