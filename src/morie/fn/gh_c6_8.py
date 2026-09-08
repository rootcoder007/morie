# morie.fn -- function file (rootcoder007/morie)
"""Weak consistency of tail-free priors.

Implements Theorem 6.27 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_tailfree_con"]


def ghosal_tailfree_con(theta0=(0.1, 0.4, 0.3, 0.2),
                        ns=(40, 160, 640), seed=42):
    """Tail-free priors are weakly consistent at every P0
    (Thm 6.27): the cell-probability posterior is a finite
    multinomial-Dirichlet problem, whose posterior mean tends to the
    true cell vector. Keys: estimate."""
    p0 = _bnp.normalize_weights(theta0)
    rng = np.random.default_rng(seed)
    errs = []
    for n in ns:
        counts = [0] * len(p0)
        for _ in range(n):
            u = float(rng.uniform(0, 1))
            acc = 0.0
            for i, q in enumerate(p0):
                acc += q
                if u <= acc:
                    counts[i] += 1
                    break
        post = [(1.0 + c) / (len(p0) + n) for c in counts]
        errs.append(max(abs(a - b) for a, b in zip(post, p0)))
    res = RichResult(payload={"estimate": errs[-1],
                              "sup_error_by_n": errs,
                              "improving": errs[-1] < errs[0],
                              "method": "tail-free multinomial reduction (GvdV 2017 Thm 6.27)"})
    return with_describe_pointer(res, "gh_c6_8")


def cheatsheet():
    return "gh_c6_8: Weak consistency of tail-free priors"
