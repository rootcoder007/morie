# morie.fn -- function file (rootcoder007/morie)
"""Mixture-of-kernels density prior.

Implements sec. 2.3.3 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_mixture_basis_prior"]


def ghosal_mixture_basis_prior(x, K=5, seed=42, bandwidth=0.15):
    """f = sum_k w_k K(x; theta_k) with Dirichlet weights (GvdV 2017
    sec. 2.3.3); Gaussian kernels at uniform random locations."""
    import math
    xs = _bnp._flat(x)
    rng = np.random.default_rng(seed)
    g = [float(rng.gamma(1.0, 1.0)) for _ in range(K)]
    w = _bnp.normalize_weights(g)
    th = [float(v) for v in rng.uniform(0, 1, K)._flat()]
    c = 1.0 / (bandwidth * math.sqrt(2.0 * math.pi))
    dens = [sum(w[k] * c * math.exp(-0.5 * ((v - th[k]) / bandwidth) ** 2)
                for k in range(K)) for v in xs]
    res = RichResult(payload={"estimate": sum(dens) / len(dens),
                              "density": dens, "weights": w,
                              "locations": th,
                              "method": "Dirichlet kernel mixture (GvdV 2017 sec. 2.3.3)"})
    return with_describe_pointer(res, "gh_c2_6")


def cheatsheet():
    return "gh_c2_6: Mixture-of-kernels density prior"
