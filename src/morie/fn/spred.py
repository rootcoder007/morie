# morie.fn -- function file (rootcoder007/morie)
"""Level-2 shrinkage predictor for cluster means."""

from math import fsum, sqrt

from ._richresult import RichResult
from ._spx import vec

__all__ = [
    "shrinkage_predictor_level2",
    "shrinkpred",
]


def shrinkage_predictor_level2(y, cluster, sigma2_u, sigma2_e):
    """Shrink cluster means toward the grand mean by their reliability.

    NOT IN SCHABENBERGER & GOTWAY -- a fixed-string search for "shrinkage"
    returns nothing. This is the empirical-Bayes / BLUP predictor of a
    random cluster effect in the two-level model

        y_ij = mu + u_j + e_ij,   u_j ~ (0, sigma2_u),
                                  e_ij ~ (0, sigma2_e),

    whose conditional expectation of the jth cluster mean is

        thetahat_j = ybar.. + (1 - lambda_j)(ybar_j - ybar..),
        lambda_j   = sigma2_e / (sigma2_e + n_j sigma2_u).

    lambda_j is the SHRINKAGE FACTOR and n_j is the cluster's own size, so
    small clusters are pulled hard toward the grand mean and large ones
    barely move. Using a common lambda for all clusters -- the usual
    error -- over-shrinks the large clusters and under-shrinks the small.

    References: Stein, C. (1956), "Inadmissibility of the usual estimator
    for the mean of a multivariate normal distribution", *Proc. 3rd
    Berkeley Symp.* 1:197-206, for the admissibility argument; Morris,
    C. N. (1983), "Parametric empirical Bayes inference: theory and
    applications", *JASA* 78:47-55, for this estimator and its variance.
    Both are named from the general literature and were NOT verified
    against a PDF in this corpus.

    The variance components are taken as GIVEN, not estimated; that is
    what "empirical Bayes alternative" means here, and the returned
    ``sigma2_u`` / ``sigma2_e`` are echoed so a caller cannot forget it.

    Parameters
    ----------
    y : (n,) array-like
        Observations.
    cluster : (n,) array-like
        Cluster codes, compared as integers.
    sigma2_u : float
        Between-cluster variance; must be non-negative.
    sigma2_e : float
        Within-cluster variance; must be positive.

    Returns
    -------
    RichResult
        ``clusters``, ``shrunk``, ``raw``, ``lambda``, ``sizes``,
        ``grand_mean``, ``n``, ``method``.
    """
    yy = vec(y, "y")
    cv = vec(cluster, "cluster")
    n = len(yy)
    if len(cv) != n:
        raise ValueError("`y` and `cluster` must have the same length")
    ci = [int(round(t)) for t in cv]
    for t, s in zip(cv, ci):
        if abs(t - s) > 1e-9:
            raise ValueError("`cluster` must hold integer codes")
    su = float(sigma2_u)
    se = float(sigma2_e)
    if su < 0:
        raise ValueError("`sigma2_u` must be non-negative")
    if se <= 0:
        raise ValueError("`sigma2_e` must be positive")

    keys = sorted(set(ci))
    if len(keys) < 2:
        raise ValueError("at least 2 clusters are needed for shrinkage "
                         "to mean anything")
    grand = fsum(yy) / n
    sizes = []
    raw = []
    lam = []
    shrunk = []
    for c in keys:
        vals = [yy[i] for i in range(n) if ci[i] == c]
        nj = float(len(vals))
        mj = fsum(vals) / nj
        lj = se / (se + nj * su)
        sizes.append(nj)
        raw.append(mj)
        lam.append(lj)
        shrunk.append(grand + (1.0 - lj) * (mj - grand))

    return RichResult(payload={
        "clusters": [float(c) for c in keys],
        "shrunk": shrunk,
        "raw": raw,
        "lambda": lam,
        "sizes": sizes,
        "grand_mean": grand,
        "sigma2_u": su,
        "sigma2_e": se,
        "shrinkage_depends_on_cluster_size": True,
        "n": n,
        "method": ("Level-2 shrinkage / empirical-Bayes predictor "
                   "(Stein 1956; Morris 1983); NOT in Schabenberger & "
                   "Gotway"),
    })


def cheatsheet():
    return "spred: level-2 shrinkage predictor for cluster means"


# compact alias per ledger/NAMING.md
shrinkpred = shrinkage_predictor_level2
