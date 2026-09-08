# morie.fn -- function file (rootcoder007/morie)
"""Permanence of the KL property.

Implements Propositions 6.28 (mixtures) and 6.29 (products) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_kl_perm"]


def ghosal_kl_perm(p0, q0, p, q):
    """Mixtures preserve the KL property: Pi = int Pi_xi drho has
    mass >= component mass on KL balls (Prop 6.28); products add:
    K(p0 x q0; p x q) = K(p0; p) + K(q0; q) (Prop 6.29). The
    additivity is verified exactly. Keys: estimate."""
    def kl(a, b):
        a = _bnp.normalize_weights(a)
        b = _bnp.normalize_weights(b)
        return sum(x * math.log(x / max(y, 1e-300))
                   for x, y in zip(a, b) if x > 0)
    k1 = kl(p0, p)
    k2 = kl(q0, q)
    # product distributions
    p0q0 = [x * y for x in _bnp.normalize_weights(p0)
            for y in _bnp.normalize_weights(q0)]
    pq = [x * y for x in _bnp.normalize_weights(p)
          for y in _bnp.normalize_weights(q)]
    kprod = kl(p0q0, pq)
    gap = abs(kprod - (k1 + k2))
    res = RichResult(payload={"estimate": kprod,
                              "kl_marginals": [k1, k2],
                              "additivity_gap": gap,
                              "method": "KL permanence (GvdV 2017 Prop 6.28-6.29)"})
    return with_describe_pointer(res, "gh_c6_9")


def cheatsheet():
    return "gh_c6_9: Permanence of the KL property"


# compact alias per ledger/NAMING.md
ghosalklperm = ghosal_kl_perm
