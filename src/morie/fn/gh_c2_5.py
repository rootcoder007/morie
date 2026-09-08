# morie.fn -- function file (rootcoder007/morie)
"""Histogram (binning) density prior.

Implements sec. 2.3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_histogram_prior"]


def ghosal_histogram_prior(x, K=8, alpha=1.0, seed=42):
    """f(x) = sum_k (p_k/|B_k|) 1{x in B_k} with (p_1..p_K) ~
    Dir(alpha,...,alpha) (GvdV 2017 sec. 2.3.2), the Dirichlet drawn
    by gamma normalization (Proposition G.2 route)."""
    xs = _bnp._flat(x)
    rng = np.random.default_rng(seed)
    g = [float(rng.gamma(alpha, 1.0)) for _ in range(K)]
    p = _bnp.normalize_weights(g)
    width = 1.0 / K
    dens = [p[min(int(v * K), K - 1)] / width
            if 0.0 <= v <= 1.0 else 0.0 for v in xs]
    res = RichResult(payload={"estimate": sum(dens) / len(dens),
                              "density": dens, "weights": p,
                              "method": "Dirichlet histogram prior (GvdV 2017 sec. 2.3.2)"})
    return with_describe_pointer(res, "gh_c2_5")


def cheatsheet():
    return "gh_c2_5: Histogram (binning) density prior"
