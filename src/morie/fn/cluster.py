# morie.fn -- function file (rootcoder007/morie)
"""One-stage cluster sampling with equal cluster sizes."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["clus1", "one_stage_cluster"]


def clus1(Y, M=float("inf"), level=0.95):
    """One-stage cluster sample: every element of each sampled cluster.

    In one-stage cluster sampling the CLUSTER is the sampling unit, so
    the variance depends on the spread of the cluster means and not at
    all on the spread within a cluster -- which is why sampling whole
    city blocks is so much less precise than sampling the same number
    of households at random.  The design effect against simple random
    sampling of the same number of elements is returned so that loss is
    a number.

    Formula: ybar_c = (1/m) sum_i ybar_i;
             v(ybar_c) = (1 - m/M) s_b^2 / m,
             s_b^2 = sum (ybar_i - ybar_c)^2 / (m - 1)

    Parameters
    ----------
    Y : array-like, shape (m, k)
        Sampled clusters, one row per cluster, all of size k.
    M : float
        Number of clusters in the population; math.inf if unknown.
    level : float
        Confidence level.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``cluster_mean``, ``between_var``, ``within_var``, ``deff``,
        ``rho``, ``m``, ``k``.

    References
    ----------
    Cochran (1977), Sampling Techniques, 3rd edition, Chapter 9, in
    which clusters of equal size are the sampling units and the
    estimate is the unweighted mean of the cluster means, with
    variance governed by the between-cluster mean square.  Chapter 9
    was NOT in the scanned excerpt available to this batch, so the
    standard published form is used; the finite-population factor
    (M - m)/M matches the convention of the sibling Cochran modules.
    """
    Y = C.mat(Y)
    m = len(Y)
    if m < 2:
        raise ValueError("at least two clusters are needed for a variance")
    k = len(Y[0])
    if any(len(r) != k for r in Y):
        raise ValueError("all clusters must have the same size")
    if k < 1:
        raise ValueError("clusters must be non-empty")
    cm = [sum(r) / k for r in Y]
    est = sum(cm) / m
    sb2 = C.var(cm, 1)
    M = float(M)
    fpc = 1.0 if math.isinf(M) else (M - m) / M
    var = fpc * sb2 / m
    se = math.sqrt(var)
    flat = [v for r in Y for v in r]
    S2 = C.var(flat, 1)
    within = (sum(C.var(r, 1) for r in Y) / m) if k > 1 else 0.0
    vsrs = S2 / (m * k)
    deff = var / vsrs if vsrs > 0 else float("nan")
    rho = (deff - 1.0) / (k - 1) if k > 1 else float("nan")
    z = C.qnorm((1.0 + float(level)) / 2.0)
    return RichResult(payload={
        "estimate": est, "se": se, "ci_lower": est - z * se,
        "ci_upper": est + z * se, "cluster_mean": cm, "between_var": sb2,
        "within_var": within, "deff": deff, "rho": rho, "m": m, "k": k,
        "method": "One-stage cluster sampling, equal cluster sizes"})


one_stage_cluster = clus1


def cheatsheet():
    return "cluster: ybar_c = mean of cluster means; v = (1-m/M) s_b^2/m"
