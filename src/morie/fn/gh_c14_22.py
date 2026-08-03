# morie.fn -- function file (rootcoder007/morie)
"""Nested Dirichlet process.

Implements sec. 14.9.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_nested_dp"]


def ghosal_nested_dp(n_groups=6, gamma=1.0, alpha=2.0, seed=42):
    """G_j | G0 ~ DP(alpha, G0) with G0 ~ DP(gamma, H) itself atomic
    ON DISTRIBUTIONS: groups cluster into identical distributions
    with positive probability (sec. 14.9.5). CRP at the top level
    over group-distributions. Keys: estimate."""
    rng = np.random.default_rng(seed)
    labels = []
    clusters = []
    for j in range(int(n_groups)):
        u = float(rng.uniform(0, 1)) * (gamma + j)
        if u < gamma:
            clusters.append(1)
            labels.append(len(clusters) - 1)
        else:
            acc = gamma
            for c in range(len(clusters)):
                acc += clusters[c]
                if u < acc:
                    clusters[c] += 1
                    labels.append(c)
                    break
    ties = int(n_groups) - len(clusters)
    res = RichResult(payload={"estimate": float(len(clusters)),
                              "group_labels": labels,
                              "groups_share_distributions":
                                  ties >= 0,
                              "method": "nested DP (GvdV 2017 sec. 14.9.5)"})
    return with_describe_pointer(res, "gh_c14_22")


def cheatsheet():
    return "gh_c14_22: Nested Dirichlet process"
