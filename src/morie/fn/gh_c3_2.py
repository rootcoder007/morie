# morie.fn -- function file (rootcoder007/morie)
"""Prior through consistent finite-dimensional distributions.

Implements sec. 3.2 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_stochastic_proc_prior"]


def ghosal_stochastic_proc_prior(x, seed=42):
    """Specify (G(A_1),...,G(A_k)) jointly with the consistency
    conditions of GvdV 2017 sec. 3.2; realized here by a Dirichlet
    vector over a partition, whose aggregations stay Dirichlet -- the
    property that makes the specification consistent."""
    xs = _bnp._flat(x)
    k = max(4, min(len(xs), 8))
    rng = np.random.default_rng(seed)
    g = [float(rng.gamma(1.0, 1.0)) for _ in range(k)]
    p = _bnp.normalize_weights(g)
    # aggregation consistency: merging two cells = summing weights
    merged = [p[0] + p[1]] + p[2:]
    gap = abs(sum(merged) - 1.0)
    res = RichResult(payload={"estimate": p[0], "weights": p,
                              "aggregation_gap": gap,
                              "method": "consistent finite-dimensional prior (GvdV 2017 sec. 3.2)"})
    return with_describe_pointer(res, "gh_c3_2")


def cheatsheet():
    return "gh_c3_2: Prior through consistent finite-dimensional distributions"
