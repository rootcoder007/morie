# morie.fn -- function file (rootcoder007/morie)
"""Empirical Bayes shrinkage estimator for cluster means."""

import numpy as np

from ._richresult import RichResult

__all__ = ["empirical_bayes_shrinkage"]


def empirical_bayes_shrinkage(y, cluster, sigma2_u=None, sigma2_e=None):
    r"""Empirical-Bayes (James-Stein) shrinkage of cluster means:

    .. math:: \hat\theta_j^{EB} = \lambda_j\bar y_j
              + (1-\lambda_j)\bar y_{\cdot\cdot},
              \qquad
              \lambda_j = \frac{\sigma_u^2}
                                {\sigma_u^2 + \sigma_e^2/n_j}.

    Each cluster mean is pulled toward the grand mean by an amount
    that depends on its OWN sample size: small clusters, whose means
    are noisy, are shrunk hard; large ones are barely moved. That
    differential is the estimator's whole content, and it is why it
    beats the raw means in total squared error -- the classic
    James-Stein result.

    The price is per-cluster BIAS. Any individual cluster's estimate
    is biased toward the middle, so the estimator is right for
    ranking and for total error and wrong for a caller who needs an
    unbiased figure for one named cluster. Both the shrinkage factors
    and the raw means are returned so that trade is inspectable.

    Parameters
    ----------
    y : array-like
        Element-level values.
    cluster : array-like
        Cluster identifiers.
    sigma2_u : float, optional
        Between-cluster variance; estimated by ANOVA otherwise.
    sigma2_e : float, optional
        Within-cluster variance; estimated otherwise.

    Returns
    -------
    RichResult
        keys: ``clusters``, ``raw_means``, ``shrunk``, ``lambda``,
        ``grand_mean``, ``sigma2_u``, ``sigma2_e``, ``n_j``,
        ``biased_per_cluster`` (True), ``method``.
    """
    yv = np.asarray(y, dtype=float).ravel()
    cl = np.asarray(cluster).ravel()
    if cl.size != yv.size:
        raise ValueError(f"cluster has {cl.size} entries for {yv.size} of y.")
    labs = np.unique(cl)
    J = labs.size
    if J < 3:
        raise ValueError(f"shrinkage needs at least 3 clusters, got {J}.")
    means = np.array([yv[cl == l].mean() for l in labs])
    nj = np.array([np.sum(cl == l) for l in labs], dtype=float)
    grand = float(np.average(means, weights=nj))
    if sigma2_e is None:
        ssw = float(np.sum([np.sum((yv[cl == l] - means[i]) ** 2)
                            for i, l in enumerate(labs)]))
        s2e = ssw / max(yv.size - J, 1)
    else:
        s2e = float(sigma2_e)
    if sigma2_u is None:
        between = float(np.var(means, ddof=1))
        s2u = max(between - s2e / float(np.mean(nj)), 0.0)
    else:
        s2u = float(sigma2_u)
    if s2e < 0 or s2u < 0:
        raise ValueError("variance components must be non-negative.")
    denom = s2u + s2e / nj
    lam = np.where(denom > 0, s2u / denom, 0.0)
    return RichResult(payload={
        "clusters": labs, "raw_means": means,
        "shrunk": lam * means + (1 - lam) * grand,
        "lambda": lam, "grand_mean": grand,
        "sigma2_u": s2u, "sigma2_e": s2e, "n_j": nj.astype(int),
        "biased_per_cluster": True,
        "tradeoff": "lower TOTAL squared error, higher bias for any single "
                    "named cluster; small clusters are shrunk hardest",
        "method": "Empirical Bayes shrinkage; lambda_j depends on the cluster's OWN size"})


def cheatsheet():
    return "ebayes: right for ranking and total error, wrong if one named cluster must be unbiased"
