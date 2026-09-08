# morie.fn -- function file (rootcoder007/morie)
r"""Causal inference in longitudinal network-dependent data.

Almost all causal inference assumes :math:`n` independent units that
are not causally connected: the intervention on one unit cannot affect
another's outcome, so the causal model only has to describe relations
*within* a unit, and inference rests on :math:`n` independent
realisations. In many cluster trials and in observational studies of a
few communities, the number of independent units is simply not large
enough for that limit to mean anything.

**The extreme case is the one worth stating.** One community of
causally connected individuals. Can the effect of a
community-level intervention on a community-level outcome -- the
average of individual outcomes, say -- still be evaluated? The chapter's
answer is yes, but only if the causal model covers **all** units at
once and identifiability is established *without* appealing to
asymptotics in a number of independent units.

**Where the replication comes from instead.** Individuals within the
community, whose dependence is restricted by the known network: unit
:math:`i` is influenced only by its friends :math:`F_i`, and
:math:`|F_i|/N \to 0`. That is what makes an average over individuals
behave like an average over weakly dependent terms, so a central limit
theorem applies in :math:`N` even with a single community. The
condition is a real one and can fail: a hub connected to a constant
fraction of the network breaks it, and ``network_summary`` reports the
maximum degree share for exactly that reason.

**Longitudinally**, the dependence compounds: a unit's covariates at
time :math:`t` may respond to its friends' treatments at
:math:`t-1`. The intervention is on the whole network at each time,
and the estimand is the mean community outcome under that policy.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 20 (Sofrygin
& van der Laan): existing causal inference assumes n independent,
causally unconnected units, so causal models need only describe
relations within a unit and inference rests on n independent
realisations; in many cluster randomized trials or observational
studies of few communities the number of independent units is not
large enough for limit-distribution inference; the extreme case of a
single community of causally connected individuals and whether a
community-level intervention's effect on a community-level outcome can
still be evaluated; and the requirement that causal models incorporate
all units and that identifiability be established under minimal
assumptions WITHOUT relying on asymptotics in a number of independent
units. Chap. 21 (the restriction of dependence to the known network
and the condition that |F_i|/N vanishes).

Sofrygin, O. & van der Laan, M. J. (2017) "Semi-Parametric Estimation
and Inference for the Mean Outcome of the Single Time-Point
Intervention in a Causally Connected Population", *Journal of Causal
Inference* 5(1), 20160003, doi:10.1515/jci-2016-0003.

Ogburn, E. L. & VanderWeele, T. J. (2014) "Causal Diagrams for
Interference", *Statistical Science* 29(4), 559-578,
doi:10.1214/14-STS501.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["network_summary", "exposure_summary",
           "community_estimand", "network_variance",
           "longitudinal_network_gcomp"]

_EPS = 1e-12


def network_summary(friends):
    r"""Degrees and the condition :math:`\max_i |F_i| / N \to 0`.

    A hub connected to a constant fraction of the network violates it,
    and then no amount of :math:`N` delivers a central limit theorem.
    """
    N = len(friends)
    if N < 2:
        raise ValueError("tlnetlg: at least 2 units are needed")
    deg = [len(set(friends[i]) - {i}) for i in range(N)]
    mx = max(deg)
    return {"N": N, "degrees": deg, "max_degree": mx,
            "mean_degree": sum(deg) / float(N),
            "max_share": mx / float(N),
            "sparse": mx / float(N) < 0.25,
            "note": "dependence must be limited to the known network "
                    "and |F_i|/N must vanish"}


def exposure_summary(A, friends, kind="fraction"):
    r"""A unit's own treatment plus a summary of its friends'.

    Reducing the friends' treatments to a fixed-dimensional summary is
    what keeps the model estimable -- the alternative is a separate
    parameter for every configuration of the neighbourhood.
    """
    a = [float(v) for v in k.vec(A)]
    N = len(a)
    if len(friends) != N:
        raise ValueError("tlnetlg: %d treatments but %d friend sets"
                         % (N, len(friends)))
    out = []
    for i in range(N):
        f = sorted(set(friends[i]) - {i})
        if not f:
            s = 0.0
        elif kind == "fraction":
            s = sum(a[j] for j in f) / len(f)
        elif kind == "count":
            s = sum(a[j] for j in f)
        elif kind == "any":
            s = 1.0 if any(a[j] == 1.0 for j in f) else 0.0
        else:
            raise ValueError("tlnetlg: kind must be fraction, count "
                             "or any, got %r" % (kind,))
        out.append((a[i], s))
    return {"summary": out, "kind": kind,
            "note": "own treatment plus a fixed-dimensional summary "
                    "of the friends'"}


def community_estimand(Q_fn, friends, W, policy):
    r"""The mean community outcome under a network-wide policy.

    ``policy(i, W)`` assigns unit :math:`i`'s treatment; the outcome
    regression then sees both own and neighbourhood exposure.
    """
    rows = [[float(v) for v in r] for r in k.mat(W)]
    N = len(rows)
    if len(friends) != N:
        raise ValueError("tlnetlg: %d covariate rows but %d friend "
                         "sets" % (N, len(friends)))
    a = [float(policy(i, rows)) for i in range(N)]
    es = exposure_summary(a, friends)["summary"]
    vals = [float(Q_fn(es[i][0], es[i][1], rows[i]))
            for i in range(N)]
    return {"psi": sum(vals) / N, "assigned": a,
            "individual": vals, "N": N}


def network_variance(ic, friends):
    r"""Variance accounting for dependence along the network.

    Sums the covariance terms for connected pairs, since exactly those
    are the ones that need not vanish. Ignoring them is the error the
    chapter's setting exists to avoid.
    """
    v = [float(q) for q in k.vec(ic)]
    N = len(v)
    if len(friends) != N:
        raise ValueError("tlnetlg: %d influence values but %d friend "
                         "sets" % (N, len(friends)))
    m = sum(v) / N
    var = sum((q - m) ** 2 for q in v) / N
    cov = 0.0
    pairs = 0
    for i in range(N):
        for j in set(friends[i]) - {i}:
            cov += (v[i] - m) * (v[j] - m)
            pairs += 1
    total = (var + cov / N) / N
    naive = var / N
    return {"se": math.sqrt(max(total, 0.0)),
            "se_naive": math.sqrt(naive),
            "n_dependent_pairs": pairs,
            "ratio": math.sqrt(max(total, 0.0) / naive)
            if naive > _EPS else float("nan"),
            "note": "only CONNECTED pairs contribute covariance; "
                    "treating units as independent drops them"}


def longitudinal_network_gcomp(Q_seq, friends, W, policy, T):
    r"""Sequential g-computation over a network, time by time.

    At each time the policy is applied to the whole network, so a
    unit's covariates may respond to its friends' earlier treatments.
    """
    if int(T) < 1:
        raise ValueError("tlnetlg: need at least one time point")
    if len(Q_seq) != int(T):
        raise ValueError("tlnetlg: %d regressions for %d time points"
                         % (len(Q_seq), T))
    rows = [[float(v) for v in r] for r in k.mat(W)]
    cur = rows
    path = []
    for t in range(int(T)):
        r = community_estimand(Q_seq[t], friends, cur, policy)
        path.append(r["psi"])
        cur = [[r["individual"][i]] + list(cur[i])
               for i in range(len(cur))]
    return RichResult(payload={
        "estimate": path[-1], "psi": path[-1], "path": path,
        "T": int(T), "network": network_summary(friends),
        "method": "longitudinal network g-computation; van der Laan & "
                  "Rose (2018) Chap. 20",
        "note": "replication comes from weakly dependent INDIVIDUALS, "
                "not from independent communities",
    })


def cheatsheet():
    return ("tlnetlg: standard causal inference assumes n independent, "
            "causally unconnected units -- useless when you observe "
            "ONE community of connected individuals. Model ALL units "
            "jointly and establish identifiability WITHOUT asymptotics "
            "in independent units. Replication comes from individuals "
            "whose dependence is restricted to the known network, with "
            "|F_i|/N -> 0; a hub connected to a constant fraction "
            "breaks that and no N repairs it. Variance must add the "
            "covariance of CONNECTED pairs.")


# compact alias per ledger/NAMING.md
networklongitudinal = longitudinal_network_gcomp
