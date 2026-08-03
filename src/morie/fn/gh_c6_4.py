# morie.fn -- function file (rootcoder007/morie)
"""Diaconis-Freedman inconsistency.

Implements Example 6.13 (mixture of DPs) of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_df_inconsist"]


def ghosal_df_inconsist(n=400, seed=42):
    """Pi = (1/2) DP(alpha) + (1/2) delta_phi with alpha_i = 2^-i and
    phi_i propto 1/(i log^2 i): the posterior contracts to phi
    whenever KL(theta0; phi) < infty even if theta0 != phi
    (Example 6.13). The demonstration computes the log posterior
    odds of the delta component, which grow linearly because the
    DP-marginal urn likelihood decays faster than phi's iid
    likelihood on thick-tailed data. Keys: estimate."""
    rng = np.random.default_rng(seed)
    phi_raw = [1.0 / (i * math.log(i + 1.0) ** 2)
               for i in range(2, 40)]
    tot = sum(phi_raw)
    phi = [v / tot for v in phi_raw]
    theta0 = phi[:]                       # truth = phi here: odds grow
    data = []
    for _ in range(n):
        u = float(rng.uniform(0, 1))
        acc = 0.0
        for i, p in enumerate(theta0):
            acc += p
            if u <= acc:
                data.append(i)
                break
        else:
            data.append(len(theta0) - 1)
    # delta-component log likelihood
    ll_phi = sum(math.log(phi[x]) for x in data)
    # DP(alpha) marginal via the Polya urn (4.13), alpha_i = 2^-i
    M_tot = sum(2.0 ** (-(i + 1)) for i in range(len(phi)))
    counts = {}
    ll_dp = 0.0
    for j, x in enumerate(data):
        a_x = 2.0 ** (-(x + 1))
        ll_dp += math.log((a_x + counts.get(x, 0))
                          / (M_tot + j))
        counts[x] = counts.get(x, 0) + 1
    log_odds = ll_phi - ll_dp             # prior odds 1:1
    res = RichResult(payload={"estimate": log_odds,
                              "delta_component_wins": log_odds > 0,
                              "method": "DF inconsistency mechanism (GvdV 2017 Ex 6.13)"})
    return with_describe_pointer(res, "gh_c6_4")


def cheatsheet():
    return "gh_c6_4: Diaconis-Freedman inconsistency"
