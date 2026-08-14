# morie.fn -- function file (rootcoder007/morie)
r"""Single time point interventions in network-dependent data.

:math:`N` units connected by a social network. For each we record
baseline covariates :math:`W_i`, exposure :math:`A_i` and outcome
:math:`Y_i`, and we observe :math:`F_i` -- the units that could
influence :math:`i`, "i's friends". The number of friends varies with
:math:`i` and is assumed to vanish relative to :math:`N`.

**Two dependencies are allowed, and naming them is the modelling
step.** A unit's *exposure* may depend on its own baseline covariates
and on those of its friends; a unit's *outcome* may depend on its own
baseline and exposure and on those of its friends. Everything else is
excluded by assumption: **all** dependence between units is fully
described by the known network. That assumption is what makes the
problem tractable, and it is also the one most likely to be wrong --
an unobserved edge is indistinguishable from unmeasured confounding
between the two units it should have joined.

**Interference means the estimand must be a policy, not a value.**
Under interference :math:`Y_i` depends on the treatments of others, so
"the effect of treatment" is not defined until the whole assignment is
specified. The estimand is the mean outcome under a stochastic policy
applied network-wide, and useful contrasts fall out of it: the
**direct** effect fixes the neighbourhood exposure and varies the
unit's own; the **spillover** effect fixes the unit's own and varies
the neighbourhood's. ``decompose_effects`` computes both, since
reporting only the total hides which mechanism produced it.

**Inference is in :math:`N` with dependence.** The influence curve
terms are correlated exactly along network edges, so the variance adds
those covariances; with :math:`\max_i|F_i|/N \to 0` the sum is still
:math:`O(N)` and a central limit theorem applies.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 21 (Sofrygin,
Ogburn & van der Laan): N units connected by a social network with
baseline covariates, exposure and outcome recorded for each, and the
observed set F_i of units connected to and able to influence i; the
number of friends varying in i and assumed to vanish when scaled by
1/N; the two permitted between-unit dependencies -- exposure depending
on own and friends' baseline covariates, outcome depending on own and
friends' baseline and exposure -- and the modelling assumption that
ANY dependence among units is fully described by the known network,
with i's exposure and outcome depending on others only through i's
friends.

Sofrygin, O. & van der Laan, M. J. (2017) "Semi-Parametric Estimation
and Inference for the Mean Outcome of the Single Time-Point
Intervention in a Causally Connected Population", *Journal of Causal
Inference* 5(1), 20160003, doi:10.1515/jci-2016-0003.

Hudgens, M. G. & Halloran, M. E. (2008) "Toward Causal Inference With
Interference", *Journal of the American Statistical Association*
103(482), 832-842, doi:10.1198/016214508000000292. Direct and
spillover effects under interference.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["friend_summary", "policy_mean", "decompose_effects",
           "network_influence_variance", "check_network_assumption"]

_EPS = 1e-12


def friend_summary(values, friends, kind="fraction"):
    r"""Summarise a quantity over :math:`F_i`."""
    v = [float(q) for q in k.vec(values)]
    N = len(v)
    if len(friends) != N:
        raise ValueError("tlnet1: %d values but %d friend sets"
                         % (N, len(friends)))
    out = []
    for i in range(N):
        f = sorted(set(friends[i]) - {i})
        if not f:
            out.append(0.0)
        elif kind == "fraction":
            out.append(sum(v[j] for j in f) / len(f))
        elif kind == "count":
            out.append(float(sum(v[j] for j in f)))
        else:
            raise ValueError("tlnet1: kind must be fraction or count, "
                             "got %r" % (kind,))
    return out


def check_network_assumption(friends, N=None):
    r"""The conditions the identification rests on.

    Degrees must vanish relative to :math:`N`, and the network must be
    symmetric -- an asymmetric "friend" relation means influence
    flows somewhere the model does not represent.
    """
    n = len(friends) if N is None else int(N)
    deg = [len(set(friends[i]) - {i}) for i in range(len(friends))]
    asym = []
    for i in range(len(friends)):
        for j in set(friends[i]) - {i}:
            if i not in set(friends[j]):
                asym.append((i, j))
    return {"max_degree": max(deg) if deg else 0,
            "max_share": (max(deg) / float(n)) if deg else 0.0,
            "sparse": (max(deg) / float(n) < 0.25) if deg else True,
            "asymmetric_edges": asym, "symmetric": not asym,
            "note": "all dependence is assumed described by the KNOWN "
                    "network; an unobserved edge is indistinguishable "
                    "from unmeasured confounding"}


def policy_mean(Q_fn, W, friends, own_prob, seed=0, draws=200):
    r"""Mean outcome under a stochastic network-wide policy.

    Treatments are drawn independently with probability
    ``own_prob``, then the neighbourhood summary follows -- so the
    estimand is a property of the POLICY, which is the only thing
    well defined under interference.
    """
    rows = [[float(v) for v in r] for r in k.mat(W)]
    N = len(rows)
    p = float(own_prob)
    if not 0.0 <= p <= 1.0:
        raise ValueError("tlnet1: the policy probability must lie in "
                         "[0,1], got %r" % (own_prob,))
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(int(draws)):
        a = [1.0 if float(rng.uniform()) < p else 0.0
             for _ in range(N)]
        fs = friend_summary(a, friends)
        tot += sum(float(Q_fn(a[i], fs[i], rows[i]))
                   for i in range(N)) / N
    return {"psi": tot / int(draws), "policy_prob": p,
            "draws": int(draws), "N": N}


def decompose_effects(Q_fn, W, friends, p_high=1.0, p_low=0.0,
                      seed=0, draws=200):
    r"""Direct and spillover effects, separately.

    Direct: own exposure varies with the neighbourhood held at
    ``p_low``. Spillover: own held at ``p_low`` while the
    neighbourhood varies. Reporting only the total hides which
    mechanism produced it.
    """
    rows = [[float(v) for v in r] for r in k.mat(W)]
    N = len(rows)
    rng = np.random.default_rng(seed)

    def mean_with(own, neigh_p):
        tot = 0.0
        for _ in range(int(draws)):
            a = [1.0 if float(rng.uniform()) < neigh_p else 0.0
                 for _ in range(N)]
            fs = friend_summary(a, friends)
            tot += sum(float(Q_fn(own, fs[i], rows[i]))
                       for i in range(N)) / N
        return tot / int(draws)

    d = mean_with(1.0, p_low) - mean_with(0.0, p_low)
    s = mean_with(0.0, p_high) - mean_with(0.0, p_low)
    tot = mean_with(1.0, p_high) - mean_with(0.0, p_low)
    return RichResult(payload={
        "estimate": {"direct": d, "spillover": s, "total": tot},
        "direct": d, "spillover": s, "total": tot,
        "method": "direct and spillover decomposition under network "
                  "interference; van der Laan & Rose (2018) Chap. 21",
        "note": "under interference the estimand is a POLICY; 'the "
                "effect of treatment' is undefined until the whole "
                "assignment is specified",
    })


def network_influence_variance(ic, friends):
    r"""Variance with covariance along edges only."""
    v = [float(q) for q in k.vec(ic)]
    N = len(v)
    if len(friends) != N:
        raise ValueError("tlnet1: %d influence values but %d friend "
                         "sets" % (N, len(friends)))
    m = sum(v) / N
    var = sum((q - m) ** 2 for q in v) / N
    cov = sum((v[i] - m) * (v[j] - m) for i in range(N)
              for j in set(friends[i]) - {i})
    tot = max((var + cov / N) / N, 0.0)
    return {"se": math.sqrt(tot),
            "se_independent": math.sqrt(var / N),
            "edges_counted": sum(len(set(friends[i]) - {i})
                                 for i in range(N)),
            "note": "correlation exists exactly along edges"}


def cheatsheet():
    return ("tlnet1: N units on a known social network, F_i = i's "
            "friends, |F_i|/N -> 0. Two dependencies allowed: exposure "
            "on own and friends' covariates, outcome on own and "
            "friends' covariates and exposures -- and ALL dependence "
            "is assumed described by the KNOWN network, which is the "
            "assumption most likely to fail, since an unobserved edge "
            "looks exactly like unmeasured confounding. Under "
            "interference the estimand must be a POLICY; direct "
            "(own exposure varies) and SPILLOVER (neighbours' varies) "
            "effects are reported separately.")


# compact alias per ledger/NAMING.md
networksingletimepoint = policy_mean
