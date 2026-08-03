# morie.fn -- function file (rootcoder007/morie)
"""Tail-free process definition.

Implements sec. 3.6, Definition 3.11 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_tailfree_def"]


def ghosal_tailfree_def(x, depth=8, seed=42):
    """A tail-free process: splitting variables independent ACROSS
    levels (GvdV 2017 Definition 3.11, p.40). Constructed with
    independent Beta splits; Proposition 3.12(i) is then checked:
    E P(A_eps) = prod_j E V (here 1/2 per level, so 2^-m)."""
    rng = np.random.default_rng(seed)
    # mass of the leftmost cell at each level, averaged over draws
    reps = 400
    means = [0.0] * depth
    for _ in range(reps):
        m = 1.0
        for lev in range(depth):
            m *= float(rng.beta(1.0, 1.0))
            means[lev] += m
    means = [v / reps for v in means]
    gaps = [abs(means[m] - 2.0 ** (-(m + 1))) for m in range(depth)]
    res = RichResult(payload={"estimate": means[-1],
                              "mean_by_level": means,
                              "prop312_gap": max(gaps),
                              "method": "tail-free splits + Prop 3.12(i) check (GvdV 2017 sec. 3.6)"})
    return with_describe_pointer(res, "gh_c3_11")


def cheatsheet():
    return "gh_c3_11: Tail-free process definition"
