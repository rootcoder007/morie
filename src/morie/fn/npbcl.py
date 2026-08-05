# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric Bayes clustering: MAP partition of a DP mixture."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["np_bayes_clustering"]


def np_bayes_clustering(y, alpha=1.0, sigma=1.0, m0=None, tau0=10.0):
    """Posterior-mode partition of a Dirichlet-process normal mixture.

    Quintana's predictive view is that a DP mixture is a rule for
    deciding, one observation at a time, whether the next value belongs
    with something already seen or starts a new group.  Reading the
    Polya urn as an objective rather than as a sampler gives a partition
    directly, without a Markov chain -- the sequential MAP allocation
    implemented here.  Observation ``i`` joins the cluster maximizing

        log n_c + log N(y_i | mu_c_post, s2_c_pred)      (existing c),
        log alpha + log N(y_i | m0, tau0^2 + sigma^2)    (new cluster),

    where, with base prior ``N(m0, tau0^2)`` and known within-cluster
    variance ``sigma^2``, the conjugate posterior for a cluster holding
    ``n_c`` values summing to ``S_c`` has precision
    ``1/tau0^2 + n_c/sigma^2`` and the predictive adds ``sigma^2``.

    Determinism: no sampling anywhere.  The allocation depends on the
    order of ``y``, which is a property of the greedy rule and is
    documented rather than hidden -- the returned ``log_score`` lets a
    caller compare orderings.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observations.
    alpha : float, default 1.0
        DP concentration; larger values favour more clusters.
    sigma : float, default 1.0
        Within-cluster standard deviation, positive.
    m0 : float or None
        Base-measure mean; the sample mean if ``None``.
    tau0 : float, default 10.0
        Base-measure standard deviation, positive.

    Returns
    -------
    RichResult
        ``labels`` (0-based cluster index per observation),
        ``estimate`` (number of clusters), ``n_clusters``, ``sizes``,
        ``means`` (posterior cluster means), ``log_score``, ``alpha``,
        ``n``.

    References
    ----------
    Quintana, F. A. (2006).  A predictive view of Bayesian clustering.
    Journal of Statistical Planning and Inference, 136(8), 2407--2429.
    doi:10.1016/j.jspi.2004.09.015
    """
    v = C.vec(y)
    n = len(v)
    if n == 0:
        raise ValueError("np_bayes_clustering: y is empty")
    a = float(alpha)
    if a <= 0.0:
        raise ValueError("np_bayes_clustering: alpha must be positive")
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("np_bayes_clustering: sigma must be positive")
    t0 = float(tau0)
    if t0 <= 0.0:
        raise ValueError("np_bayes_clustering: tau0 must be positive")
    mu0 = C.mean(v) if m0 is None else float(m0)
    s2 = s * s
    p0 = 1.0 / (t0 * t0)

    def lnorm(x, mu, var):
        return -0.5 * (math.log(2.0 * math.pi * var) + (x - mu) ** 2 / var)

    labels = []
    counts = []
    sums = []
    total = 0.0
    for i in range(n):
        best = None
        bestk = -1
        for c in range(len(counts)):
            prec = p0 + counts[c] / s2
            mc = (mu0 * p0 + sums[c] / s2) / prec
            sc = 1.0 / prec + s2
            sco = math.log(counts[c]) + lnorm(v[i], mc, sc)
            if best is None or sco > best:
                best = sco
                bestk = c
        newsco = math.log(a) + lnorm(v[i], mu0, t0 * t0 + s2)
        if best is None or newsco > best:
            labels.append(len(counts))
            counts.append(1)
            sums.append(v[i])
            total += newsco
        else:
            labels.append(bestk)
            counts[bestk] += 1
            sums[bestk] += v[i]
            total += best
    means = []
    for c in range(len(counts)):
        prec = p0 + counts[c] / s2
        means.append((mu0 * p0 + sums[c] / s2) / prec)
    return RichResult(payload={
        "labels": labels, "estimate": float(len(counts)),
        "n_clusters": len(counts), "sizes": [float(x) for x in counts],
        "means": means, "log_score": total, "alpha": a, "n": n,
        "method": "DP-mixture MAP partition (Quintana 2006)"})


def cheatsheet():
    return "npbcl: MAP partition of a Dirichlet-process mixture"


npbayesclustering = np_bayes_clustering
