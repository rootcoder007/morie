# morie.fn -- function file (rootcoder007/morie)
"""Random basis expansion prior.

Implements sec. 2.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_random_basis_expansion"]


def ghosal_random_basis_expansion(x, n_terms=12, seed=42, decay=1.5):
    """f = sum_k z_k phi_k with independent z_k (GvdV 2017 sec. 2.1);
    cosine basis, z_k ~ N(0, k^-2*decay) so the series converges."""
    import math
    xs = _bnp._flat(x)
    rng = np.random.default_rng(seed)
    z = [float(v) * (k + 1.0) ** (-decay)
         for k, v in enumerate(rng.normal(0, 1, n_terms)._flat())]
    f = [sum(z[k] * math.cos(math.pi * (k + 1) * v)
             for k in range(n_terms)) for v in xs]
    est = sum(f) / len(f)
    res = RichResult(payload={"estimate": est, "f": f,
                              "coefficients": z,
                              "method": "random cosine-basis expansion (GvdV 2017 sec. 2.1)"})
    return with_describe_pointer(res, "gh_c2_1")


def cheatsheet():
    return "gh_c2_1: Random basis expansion prior"
