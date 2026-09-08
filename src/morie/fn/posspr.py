# morie.fn -- function file (rootcoder007/morie)
r"""The Pólya urn predictive, and the density it implies.

Under a Dirichlet process prior the predictive distribution of the
next latent parameter given the previous ones has no integral in it:

.. math:: \theta_{n+1}\mid\theta_1,\dots,\theta_n \;\sim\;
          \frac{\alpha}{\alpha+n}G_0
          + \frac{1}{\alpha+n}\sum_{i=1}^{n}\delta_{\theta_i}.

Everything follows from that line. The mixture has two parts -- draw
a **new** value from the base measure with probability
:math:`\alpha/(\alpha+n)`, or **repeat** an existing one with
probability proportional to how often it has already appeared -- and
because the repeat term is a sum of point masses, the DP produces
**ties with positive probability**. That is not an artefact; it is
what makes the DP a clustering prior at all, and it is why a DP
mixture is used rather than a DP directly for continuous data.

**The observable density is the mixture the urn induces.** With a
kernel :math:`k(y\mid\theta)`,

.. math:: p(y_{n+1}\mid y_{1:n}) = \frac{\alpha}{\alpha+n}
          \int k(y\mid\theta)\,dG_0(\theta)
          + \frac{1}{\alpha+n}\sum_{j} n_j\,k(y\mid\theta_j^*),

so the predictive density is a weighted sum over the **occupied
clusters** plus one term for a cluster not yet seen. The weight on
that last term is exactly :math:`\alpha/(\alpha+n)` -- the model's
stated probability that the next observation is unlike everything
seen so far, which is a quantity worth reporting rather than burying.

**Concentration is not a smoothing parameter.** :math:`\alpha`
controls how readily new clusters appear, and the expected number of
them grows like :math:`\alpha\log n`, so doubling :math:`\alpha` does
not double the clusters. ``expected_clusters`` computes it exactly.

**Two observations are tied with probability
:math:`1/(1+\alpha)`.** That closed form falls straight out of the
urn with :math:`n=1`, and the anchor checks it by simulation --
a test that fails if the two mixture weights are ever swapped.

References
----------
Muller, P. & Quintana, F. A. (2004) "Nonparametric Bayesian Data
Analysis", *Statistical Science* 19(1), 95-110,
doi:10.1214/088342304000000017. [PDF supplied by Vee.] The review of
nonparametric Bayesian inference organised by inference problem --
density estimation, regression, survival analysis, hierarchical
models and model validation -- covering DP models and variations,
Polya trees, dependent DP models, and the predictive rules used for
posterior simulation.

Blackwell, D. & MacQueen, J. B. (1973) "Ferguson Distributions via
Polya Urn Schemes", *The Annals of Statistics* 1(2), 353-355,
doi:10.1214/aos/1176342372. The urn representation itself. NOTE: not
held locally; the rule is quoted in the Muller-Quintana review and in
Li et al. (2015), both of which are.

Ferguson, T. S. (1973) "A Bayesian Analysis of Some Nonparametric
Problems", *The Annals of Statistics* 1(2), 209-230,
doi:10.1214/aos/1176342360. The prior. NOTE: not held locally.

Escobar, M. D. & West, M. (1995) "Bayesian Density Estimation and
Inference Using Mixtures", *JASA* 90(430), 577-588,
doi:10.1080/01621459.1995.10476550. The DP mixture density estimate.
NOTE: not held locally.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["urn_weights", "sample_urn", "predictive_density",
           "expected_clusters", "tie_probability"]

_EPS = 1e-12


def urn_weights(counts, alpha):
    r""":math:`(\alpha, n_1,\dots,n_K)/(\alpha+n)`.

    The new-cluster weight is :math:`\alpha/(\alpha+n)` exactly, and
    it is returned separately because it IS the model's answer to
    "how likely is something new".
    """
    c = [float(v) for v in k.vec(counts)] if list(counts) else []
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("posspr: the concentration must be "
                         "positive")
    if any(v <= 0.0 for v in c):
        raise ValueError("posspr: an occupied cluster must have a "
                         "positive count")
    n = sum(c)
    return {"existing": [v / (a + n) for v in c],
            "new": a / (a + n), "n": n, "K": len(c),
            "total": (sum(c) + a) / (a + n),
            "note": "repeat in proportion to how often it has "
                    "already appeared; that is the clustering"}


def sample_urn(n, alpha, rng=None, seed=0):
    r"""Draw a Pólya urn sequence of cluster labels."""
    a = float(alpha)
    N = int(n)
    if N < 1:
        raise ValueError("posspr: n must be at least 1")
    r = rng if rng is not None else np.random.default_rng(seed)
    counts, labels = [], []
    for _ in range(N):
        if not counts:
            counts.append(1.0)
            labels.append(0)
            continue
        w = urn_weights(counts, a)
        u = float(r.uniform())
        acc, chosen = 0.0, None
        for j in range(len(counts)):
            acc += w["existing"][j]
            if u <= acc:
                chosen = j
                break
        if chosen is None:
            counts.append(1.0)
            labels.append(len(counts) - 1)
        else:
            counts[chosen] += 1.0
            labels.append(chosen)
    return {"labels": labels, "counts": counts,
            "n_clusters": len(counts), "alpha": a,
            "note": "ties occur with POSITIVE probability, which is "
                    "why the DP clusters"}


def predictive_density(grid, cluster_params, counts, alpha, kernel,
                       base_predictive):
    r"""Occupied clusters plus one term for the unseen.

    ``base_predictive(y)`` is :math:`\int k(y\mid\theta)dG_0(\theta)`
    -- the prior predictive, which is what a brand-new cluster
    contributes.
    """
    w = urn_weights(counts, alpha)
    out, new_share = [], w["new"]
    for y in grid:
        v = new_share * float(base_predictive(y))
        for j in range(len(counts)):
            v += w["existing"][j] * float(kernel(y,
                                                 cluster_params[j]))
        out.append(v)
    return RichResult(payload={
        "estimate": out, "density": out, "grid": list(grid),
        "new_cluster_weight": new_share,
        "occupied_weights": w["existing"], "K": w["K"], "n": w["n"],
        "method": "DP mixture posterior predictive; Muller & "
                  "Quintana (2004)",
        "note": "the weight on the unseen component is exactly "
                "alpha/(alpha+n) -- report it rather than bury it",
    })


def expected_clusters(n, alpha):
    r""":math:`\sum_{i=0}^{n-1}\alpha/(\alpha+i)`, which grows like
    :math:`\alpha\log n`."""
    a = float(alpha)
    N = int(n)
    if a <= 0.0 or N < 1:
        raise ValueError("posspr: need alpha > 0 and n >= 1")
    e = sum(a / (a + i) for i in range(N))
    return {"expected": e, "n": N, "alpha": a,
            "log_approximation": a * math.log(1.0 + N / a),
            "note": "logarithmic in n, so alpha is not a smoothing "
                    "knob that scales the cluster count linearly"}


def tie_probability(alpha):
    r""":math:`P(\theta_2=\theta_1) = 1/(1+\alpha)`.

    Straight from the urn at :math:`n=1`, and a check that fails if
    the two mixture weights are ever exchanged.
    """
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("posspr: the concentration must be "
                         "positive")
    return {"tie": 1.0 / (1.0 + a), "new": a / (1.0 + a),
            "alpha": a}


def cheatsheet():
    return ("posspr: the DP predictive has NO integral -- "
            "theta_{n+1} ~ (alpha G_0 + sum delta_{theta_i})/"
            "(alpha + n). Draw NEW with probability alpha/(alpha+n) or "
            "REPEAT in proportion to how often a value has already "
            "appeared, so TIES have positive probability, which is "
            "exactly why the DP clusters and why continuous data need "
            "a DP MIXTURE rather than a DP. The predictive density is "
            "then a weighted sum over occupied clusters plus one "
            "prior-predictive term whose weight, alpha/(alpha+n), is "
            "the model's stated probability of something new. Cluster "
            "count grows like alpha log n; two draws tie with "
            "probability exactly 1/(1+alpha).")


# compact alias per ledger/NAMING.md
posterior_predictive_np = predictive_density

# public names resolved by fn/_lazy_map.json
posterior_predictive = predictive_density
