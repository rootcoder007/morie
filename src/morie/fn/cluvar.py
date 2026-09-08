# morie.fn -- function file (rootcoder007/morie)
"""Cluster sample variance estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["cluster_variance"]


def cluster_variance(y, cluster, N=None):
    r"""Variance of a mean under cluster sampling:

    .. math:: \widehat{\operatorname{Var}}(\bar y_{clu})
              = \Big(1 - \frac{n}{N}\Big)\frac{S_b^2}{n},

    where :math:`S_b^2` is the variance BETWEEN cluster means and
    :math:`n` the number of CLUSTERS -- not the number of elements.

    That substitution is the whole subject. Sampling 40 households
    from each of 25 villages gives 1000 observations and 25 degrees
    of freedom, and treating it as 1000 independent draws understates
    the standard error by the square root of the design effect,

    .. math:: \text{deff} = 1 + (\bar m - 1)\rho,

    with :math:`\rho` the intra-cluster correlation. Both are
    computed and returned, so the cost of clustering is a number:
    with :math:`\rho = 0.05` and clusters of 40 the effective sample
    size is roughly a third of the nominal one.

    Parameters
    ----------
    y : array-like
        Element-level values.
    cluster : array-like
        Cluster identifiers.
    N : int, optional
        Number of clusters in the population, for the fpc.

    Returns
    -------
    RichResult
        keys: ``mean``, ``variance``, ``se``, ``n_clusters``,
        ``n_elements``, ``mean_cluster_size``, ``icc``, ``deff``,
        ``effective_n``, ``naive_se``, ``se_inflation``, ``method``.
    """
    yv = np.asarray(y, dtype=float).ravel()
    cl = np.asarray(cluster).ravel()
    if cl.size != yv.size:
        raise ValueError(f"cluster has {cl.size} entries for {yv.size} of y.")
    labs = np.unique(cl)
    n = labs.size
    if n < 2:
        raise ValueError(f"need at least 2 clusters, got {n}.")
    means = np.array([yv[cl == l].mean() for l in labs])
    sizes = np.array([np.sum(cl == l) for l in labs], dtype=float)
    Sb2 = float(np.var(means, ddof=1))
    fpc = 1.0 if N is None else max(0.0, 1.0 - n / float(N))
    var = fpc * Sb2 / n
    mbar = float(sizes.mean())
    # one-way ANOVA intra-cluster correlation
    grand = float(yv.mean())
    ssb = float(np.sum(sizes * (means - grand) ** 2))
    ssw = float(np.sum([(np.sum((yv[cl == l] - means[i]) ** 2))
                        for i, l in enumerate(labs)]))
    msb = ssb / (n - 1)
    msw = ssw / max(yv.size - n, 1)
    icc = float((msb - msw) / (msb + (mbar - 1) * msw)) if (msb + (mbar - 1) * msw) != 0 else 0.0
    deff = 1.0 + (mbar - 1.0) * icc
    naive = float(np.std(yv, ddof=1) / np.sqrt(yv.size))
    return RichResult(payload={
        "mean": float(means.mean()), "variance": var,
        "se": float(np.sqrt(max(var, 0.0))),
        "n_clusters": int(n), "n_elements": int(yv.size),
        "mean_cluster_size": mbar, "icc": icc, "deff": float(deff),
        "effective_n": float(yv.size / deff) if deff > 0 else float(yv.size),
        "naive_se": naive,
        "se_inflation": float(np.sqrt(max(var, 0.0)) / naive) if naive > 0 else np.nan,
        "note": "n is the number of CLUSTERS, not elements",
        "method": "Cluster variance from between-cluster spread; deff = 1 + (mbar - 1) rho"})


def cheatsheet():
    return "cluvar: 1000 elements in 25 villages is 25 degrees of freedom, not 1000"
