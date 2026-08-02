# morie.fn -- function file (rootcoder007/morie)
"""Generalized estimating equations for survival."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["general_estimating_eq_surv"]


def general_estimating_eq_surv(time, event, x, cluster=None):
    r"""Cox model with a cluster-robust (GEE-style) sandwich variance.

    The point estimate is the usual partial-likelihood one; what
    changes is the variance. Under clustering -- littermates,
    repeated events on the same subject, patients within hospitals --
    the partial-likelihood information is wrong, because the
    contributions are not independent.

    The estimator is a WORKING-INDEPENDENCE one: it deliberately uses
    the independence score and repairs the variance afterwards,
    rather than modelling the dependence. That is the GEE bargain,
    and it is usually the right one here because a correct dependence
    model for recurrent event times is hard and the point estimate
    stays consistent without it.

    The correction only matters when clusters are informative:
    ``variance_inflation`` reports the ratio, and it is near one when
    they are not.

    Parameters
    ----------
    time, event : array-like
        Times and 0/1 indicators.
    x : array-like
        Covariates.
    cluster : array-like, optional
        Cluster identifiers; independence is assumed otherwise.

    Returns
    -------
    RichResult
        keys: ``beta``, ``se_model``, ``se_robust``,
        ``variance_inflation``, ``n_clusters``, ``working_model``,
        ``n_events``, ``n``, ``method``.
    """
    from .survvar import variance_cox_estimator

    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != tv.size:
        X = X.T
    if X.shape[0] != tv.size:
        raise ValueError("x must have one row per follow-up time.")
    n, p = X.shape
    if ev.sum() < 2:
        raise ValueError("need at least 2 events.")

    def score(b):
        w = np.exp(X @ b)
        g = np.zeros(p)
        for i in np.nonzero(ev == 1.0)[0]:
            at = tv >= tv[i]
            sw = float(w[at].sum())
            if sw > 0:
                g += X[i] - (w[at, None] * X[at]).sum(axis=0) / sw
        return g

    b = np.zeros(p)
    for _ in range(60):
        v = variance_cox_estimator(b, X, tv, ev)
        step = v["variance"] @ score(b)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    fit = variance_cox_estimator(b, X, tv, ev, robust=True, cluster=cluster)
    nc = None if cluster is None else int(np.unique(np.asarray(cluster)).size)
    infl = fit["ratio"]
    return RichResult(payload={
        "beta": b, "se_model": fit["se"], "se_robust": fit["robust_se"],
        "variance_inflation": infl, "n_clusters": nc,
        "working_model": "independence: the score ignores the clustering and "
                         "the variance repairs it afterwards",
        "when_it_matters": "only when clusters are informative; the inflation "
                           "is near one when they are not",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Cox point estimate with a cluster-robust GEE sandwich"})


def cheatsheet():
    return "survgen: working independence for the score, sandwich for the variance -- the GEE bargain"
