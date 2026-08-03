# morie.fn -- function file (rootcoder007/morie)
"""Completely random measure.

Implements Appendix J (definition) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP (appendices).
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_completely_random_measure"]


def ghosal_completely_random_measure(n_sim=1500, seed=42):
    """M(A) independent of M(B) for disjoint A, B (App J): gamma CRM
    on two disjoint halves -- empirical correlation ~ 0.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for _ in range(n_sim):
        xs.append(float(rng.gamma(1.0, 1.0)))
        ys.append(float(rng.gamma(2.0, 1.0)))
    mx = sum(xs) / n_sim
    my = sum(ys) / n_sim
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)
                    * sum((y - my) ** 2 for y in ys))
    corr = num / den
    res = RichResult(payload={"estimate": corr,
                              "independent": abs(corr) < 0.08,
                              "method": "CRM independence (GvdV 2017 App J)"})
    return with_describe_pointer(res, "gh_crm_def")


def cheatsheet():
    return "gh_crm_def: Completely random measure"
