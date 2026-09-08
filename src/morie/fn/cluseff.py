"""Intracluster correlation and the survey design effect."""

from . import _array_core as np
from ._richresult import RichResult

__all__ = ["intracluster_correlation_rho"]


def intracluster_correlation_rho(y, cluster):
    r"""Intracluster correlation and Kish's design effect.

    One-way random-effects decomposition of the response. With ``a``
    clusters and :math:`n_i` units in cluster ``i``,

    .. math::

       MSB = \frac{\sum_i n_i(\bar y_i-\bar y)^2}{a-1}, \qquad
       MSW = \frac{\sum_i\sum_j (y_{ij}-\bar y_i)^2}{N-a},

    and with the unequal-size correction

    .. math::

       n_0=\frac{1}{a-1}\Bigl(N-\frac{\sum_i n_i^2}{N}\Bigr), \qquad
       \hat\sigma^2_a=\frac{MSB-MSW}{n_0}, \qquad
       \rho=\frac{\hat\sigma^2_a}{\hat\sigma^2_a+MSW}.

    Kish's design effect follows as
    :math:`DEFF = 1+(\bar n-1)\rho`: the factor by which cluster
    sampling inflates the variance of the mean relative to a simple
    random sample of the same size.

    ``rho`` can come out negative -- when within-cluster spread exceeds
    between-cluster spread the ANOVA estimator goes below zero. That is
    reported as computed rather than clipped to 0, because silently
    clipping hides a badly specified clustering.

    Parameters
    ----------
    y : array-like
        Response, one value per unit.
    cluster : array-like
        Cluster label per unit; any hashable labels.

    Returns
    -------
    RichResult
        Keys ``rho``, ``deff``, ``msb``, ``msw``, ``n0``, ``var_between``,
        ``n_clusters``, ``n_obs``, ``mean_cluster_size``,
        ``effective_n`` (= N / DEFF).

    References
    ----------
    Kish, L. (1965). *Survey Sampling*. Wiley, sec. 5.4 (design effect).
    Estimator cross-checked against the reference implementation in the
    ICC R package (``ICCbare``), which uses the same ``n0`` correction.
    """
    v = [float(t) for t in np.asarray(y, dtype=float).ravel().tolist()]
    g = [t for t in np.asarray(cluster).ravel().tolist()]
    if len(v) != len(g):
        raise ValueError("y and cluster must have the same length.")
    groups = {}
    for val, lab in zip(v, g):
        groups.setdefault(lab, []).append(val)
    a = len(groups)
    n_tot = len(v)
    if a < 2:
        raise ValueError("need at least 2 clusters.")
    if n_tot <= a:
        raise ValueError("need more observations than clusters.")
    grand = sum(v) / n_tot
    sizes = [len(vals) for vals in groups.values()]
    ssb = sum(len(vals) * (sum(vals) / len(vals) - grand) ** 2 for vals in groups.values())
    ssw = sum(sum((t - sum(vals) / len(vals)) ** 2 for t in vals) for vals in groups.values())
    msb = ssb / (a - 1)
    msw = ssw / (n_tot - a)
    n0 = (n_tot - sum(s * s for s in sizes) / n_tot) / (a - 1)
    var_a = (msb - msw) / n0
    denom = var_a + msw
    if denom == 0:
        raise ValueError("zero total variance; rho is undefined.")
    rho = var_a / denom
    nbar = n_tot / a
    deff = 1.0 + (nbar - 1.0) * rho
    return RichResult(
        payload={
            "rho": float(rho),
            "deff": float(deff),
            "msb": float(msb),
            "msw": float(msw),
            "n0": float(n0),
            "var_between": float(var_a),
            "n_clusters": a,
            "n_obs": n_tot,
            "cluster_sizes": sizes,
            "mean_cluster_size": float(nbar),
            "effective_n": float(n_tot / deff) if deff > 0 else float("nan"),
            "method": "One-way ANOVA intracluster correlation; Kish design effect",
        }
    )


def cheatsheet():
    return "cluseff: intracluster correlation rho and Kish's design effect"
