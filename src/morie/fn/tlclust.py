# morie.fn -- function file (rootcoder007/morie)
r"""LTMLE with clustering.

Breastfeeding cannot be randomised. PROBIT randomised *hospitals* to a
programme that encouraged and supported it, giving indirect evidence
about its protective effect on infant infection and hospitalisation.
Two features of that design shape the estimator.

**Hospitalisation is both outcome and confounder.** The exposure is a
longitudinal one -- duration of breastfeeding -- and a hospitalisation
both counts toward the outcome and affects whether breastfeeding
continues. So it is a **time-varying confounder affected by prior
exposure**, the situation for which standard adjustment fails in both
directions: condition on it and you block part of the effect; ignore
it and confounding remains. Sequential g-computation is what handles
it.

**Two parametrizations of the g-formula, and they are not equivalent
in practice.** The outcome can be modelled directly at the end of
follow-up, or built up period by period from the time-varying
components. They identify the same quantity, but they use different
regressions and so fail differently under misspecification --
implementing only one hides that. Both are implemented, and the anchor
requires them to agree on a correctly specified model and to diverge
under a misspecified one, which is the informative comparison.

**Clustering changes the variance, not the point estimate.** Infants
within a hospital are not independent, so the influence curve must be
aggregated to the **cluster** before its variance is taken:

.. math:: \widehat{\mathrm{Var}} = \frac{1}{J}\,
          \mathrm{var}\Big(\sum_{i \in \text{cluster } j} D^*_i\Big),

with :math:`J` the number of clusters. Treating the infants as
independent understates the standard error whenever the
within-cluster correlation is positive, and the anchor measures that
understatement rather than asserting it.

References
----------
van der Laan, M. J. & Rose, S. (2018) *Targeted Learning in Data
Science*, Springer, doi:10.1007/978-3-319-65304-4. Chap. 15
(Schnitzer, van der Laan, Moodie & Platt): the PROBIT
cluster-randomized trial of a breastfeeding promotion programme, used
because breastfeeding cannot be directly allocated; the estimation of
the effect of different durations of breastfeeding on the number of
periods of hospitalization in the first year; hospitalizations treated
as a TIME-VARYING CONFOUNDER because they may also affect continuation
of breastfeeding; the two parametrizations of the g-formula; and an
LTMLE implementation accounting for an outcome partially determined by
time-varying confounders and for the clustering arising from the study
design.

Kramer, M. S. et al. (2001) "Promotion of Breastfeeding Intervention
Trial (PROBIT): a randomized trial in the Republic of Belarus",
*JAMA* 285(4), 413-420, doi:10.1001/jama.285.4.413. The trial.

Schnitzer, M. E., Moodie, E. E. M., van der Laan, M. J., Platt, R. W.
& Klein, M. B. (2014) "Modeling the impact of hepatitis C viral
clearance on end-stage liver disease in an HIV co-infected cohort with
targeted maximum likelihood estimation", *Biometrics* 70(1), 144-152,
doi:10.1111/biom.12105.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["cluster_variance", "naive_variance",
           "g_formula_pooled", "g_formula_sequential",
           "ltmle_clustered", "design_effect"]

_EPS = 1e-12


def naive_variance(ic):
    r"""The variance that treats every unit as independent."""
    v = [float(q) for q in k.vec(ic)]
    n = len(v)
    if n < 2:
        raise ValueError("tlclust: at least 2 observations are needed")
    m = sum(v) / n
    return math.sqrt(sum((q - m) ** 2 for q in v) / (n - 1) / n)


def cluster_variance(ic, cluster):
    r"""Aggregate the influence curve to the cluster, then vary.

    The cluster is the independent unit; the individual is not.
    """
    v = [float(q) for q in k.vec(ic)]
    c = list(cluster)
    if len(v) != len(c):
        raise ValueError("tlclust: %d influence values for %d cluster "
                         "labels" % (len(v), len(c)))
    agg = {}
    for i in range(len(v)):
        agg[c[i]] = agg.get(c[i], 0.0) + v[i]
    J = len(agg)
    if J < 2:
        raise ValueError("tlclust: at least 2 clusters are needed")
    sums = list(agg.values())
    m = sum(sums) / J
    var = sum((q - m) ** 2 for q in sums) / (J - 1)
    return {"se": math.sqrt(var / J) / (len(v) / float(J)),
            "n_clusters": J, "cluster_sums": sums,
            "note": "the CLUSTER is independent, not the individual"}


def design_effect(ic, cluster):
    r"""How badly the independence assumption understates the error."""
    a = naive_variance(ic)
    b = cluster_variance(ic, cluster)["se"]
    return {"se_naive": a, "se_clustered": b,
            "ratio": b / a if a > 0 else float("nan"),
            "note": "a ratio above 1 is the understatement caused by "
                    "treating within-cluster observations as "
                    "independent"}


def g_formula_pooled(Q_final, weights=None):
    r"""Parametrization 1: model the outcome directly at the end.

    One regression on the whole history -- fewer models, and every one
    of them has to be right about the whole history at once.
    """
    q = [float(v) for v in k.vec(Q_final)]
    w = [1.0] * len(q) if weights is None else [float(v)
                                                for v in k.vec(weights)]
    t = sum(w)
    if t <= _EPS:
        raise ValueError("tlclust: the weights sum to zero")
    return {"psi": sum(w[i] * q[i] for i in range(len(q))) / t,
            "parametrization": "pooled",
            "note": "one regression on the full history"}


def g_formula_sequential(Q_seq):
    r"""Parametrization 2: build the outcome up period by period.

    ``Q_seq[t]`` is the regression at time :math:`t`, fitted with the
    next period's prediction as its outcome. Identifies the same
    quantity; misspecifies differently.
    """
    if not Q_seq:
        raise ValueError("tlclust: the sequence is empty")
    cur = [float(v) for v in k.vec(Q_seq[-1])]
    for t in range(len(Q_seq) - 2, -1, -1):
        nxt = [float(v) for v in k.vec(Q_seq[t])]
        if len(nxt) != len(cur):
            raise ValueError("tlclust: the regressions differ in "
                             "length at time %d" % t)
        cur = nxt
    return {"psi": sum(cur) / len(cur),
            "parametrization": "sequential",
            "T": len(Q_seq),
            "note": "identifies the same estimand; fails differently "
                    "under misspecification"}


def ltmle_clustered(Q_seq, H_seq, Y, cluster):
    r"""LTMLE with cluster-level inference.

    The point estimate is the ordinary sequential TMLE; only the
    variance recognises the design.
    """
    from .tlltmle import ltmle
    r = ltmle(Q_seq, H_seq, Y)
    q = r["Q_star"][-1]
    psi = r["psi"]
    ic = [float(v) - psi for v in q]
    cv = cluster_variance(ic, cluster)
    nv = naive_variance(ic)
    return RichResult(payload={
        "estimate": psi, "psi": psi,
        "se_clustered": cv["se"], "se_naive": nv,
        "ci": (psi - 1.96 * cv["se"], psi + 1.96 * cv["se"]),
        "n_clusters": cv["n_clusters"],
        "design_effect": cv["se"] / nv if nv > 0 else float("nan"),
        "method": "LTMLE with cluster-level influence-curve inference; "
                  "van der Laan & Rose (2018) Chap. 15",
        "note": "clustering changes the VARIANCE, not the point "
                "estimate",
    })


def cheatsheet():
    return ("tlclust: PROBIT randomised HOSPITALS because breastfeeding "
            "cannot be allocated. Hospitalisation is both part of the "
            "outcome and a TIME-VARYING CONFOUNDER affected by prior "
            "exposure -- condition on it and you block the effect, "
            "ignore it and confounding stays; sequential "
            "g-computation is what handles it. TWO parametrizations of "
            "the g-formula identify the same estimand and misspecify "
            "differently, so implement both. Clustering changes the "
            "VARIANCE only: aggregate the influence curve to the "
            "cluster before taking its variance, or the standard error "
            "is understated.")


# compact alias per ledger/NAMING.md
clusteredltmle = ltmle_clustered
