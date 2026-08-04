# morie.fn -- function file (rootcoder007/morie)
"""Poisson-kernel Dirichlet mixture.

Implements sec. 5.5 (conjugate kernels); model eq. (5.1) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_poi_ker"]


def ghosal_poi_ker(k_values, lambdas=None, weights=None, alpha=1.0,
                   n_terms=100, seed=42):
    """f(k) = int Poi(k; lambda) dG(lambda), G ~ DP (eq. 5.1 with a
    Poisson kernel, sec. 5.5): evaluates the mixture pmf either at
    supplied atoms/weights or at a stick-breaking draw with a gamma
    center measure. Keys: estimate."""
    ks = [int(v) for v in _bnp._flat(k_values)]
    rng = np.random.default_rng(seed)
    if lambdas is None:
        M = float(alpha)
        V = [float(rng.beta(1.0, M)) for _ in range(n_terms)]
        weights = _bnp.stick_breaking(V)
        lambdas = [float(rng.gamma(2.0, 1.0)) for _ in range(n_terms)]
    else:
        lambdas = _bnp._flat(lambdas)
        weights = _bnp.normalize_weights(weights)
    def poi(k, lam):
        return math.exp(-lam + k * math.log(lam)
                        - math.lgamma(k + 1.0))
    pmf = [sum(w * poi(k, l) for w, l in zip(weights, lambdas))
           for k in ks]
    res = RichResult(payload={"estimate": pmf[0], "pmf": pmf,
                              "method": "Poisson DP mixture (GvdV 2017 sec. 5.5)"})
    return with_describe_pointer(res, "gh_c5_10")


def cheatsheet():
    return "gh_c5_10: Poisson-kernel Dirichlet mixture"


# compact alias per ledger/NAMING.md
ghosalpoiker = ghosal_poi_ker
