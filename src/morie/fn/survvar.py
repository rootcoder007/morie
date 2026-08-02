# morie.fn -- function file (rootcoder007/morie)
"""Variance estimator for Cox beta."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["variance_cox_estimator"]


def variance_cox_estimator(beta, z, time, event, robust=False, cluster=None):
    r"""Variance of the Cox coefficient from the observed information:

    .. math:: \widehat{\operatorname{Var}}(\hat\beta)
              = \mathcal I(\hat\beta)^{-1},
              \qquad
              \mathcal I(\beta) = -\frac{\partial^2 \ell_p}
                                          {\partial\beta\partial\beta'},

    with :math:`\ell_p` the partial log-likelihood.

    The partial likelihood is not a likelihood -- the baseline hazard
    has been profiled out -- yet its observed information behaves as
    if it were: that is the substantive result behind Cox regression,
    not an approximation being waved through.

    A robust (Lin-Wei) sandwich is also available and is the right
    choice under clustering or model misspecification. The two differ
    exactly when the proportional-hazards model is wrong, so
    comparing them is a specification diagnostic, and ``ratio`` is
    returned for that purpose.

    Parameters
    ----------
    beta : array-like
        Coefficients at which to evaluate.
    z : array-like
        Covariates.
    time, event : array-like
        Follow-up times and 0/1 indicators.
    robust : bool
        Return the sandwich as well.
    cluster : array-like, optional
        Cluster identifiers for the robust variance.

    Returns
    -------
    RichResult
        keys: ``information``, ``variance``, ``se``,
        ``robust_variance``, ``robust_se``, ``ratio``,
        ``partial_likelihood_note``, ``n_events``, ``n``, ``method``.
    """
    b = np.atleast_1d(np.asarray(beta, dtype=float)).ravel()
    tv = np.asarray(time, dtype=float).ravel()
    ev = np.asarray(event, dtype=float).ravel()
    Z = np.atleast_2d(np.asarray(z, dtype=float))
    if Z.shape[0] != tv.size:
        Z = Z.T
    if Z.shape[0] != tv.size:
        raise ValueError("z must have one row per follow-up time.")
    if ev.size != tv.size:
        raise ValueError(f"event has {ev.size} entries for {tv.size} times.")
    if not np.all(np.isin(ev, (0.0, 1.0))):
        raise ValueError("event must be binary 0/1.")
    n, p = Z.shape
    if b.size != p:
        raise ValueError(f"beta has {b.size} entries for {p} covariates.")
    if ev.sum() < 1:
        raise ValueError("no events: the information is zero.")
    w = np.exp(Z @ b)
    info = np.zeros((p, p))
    score_i = np.zeros((n, p))
    for i in np.nonzero(ev == 1.0)[0]:
        at = tv >= tv[i]
        sw = float(w[at].sum())
        if sw <= 0:
            continue
        zbar = (w[at, None] * Z[at]).sum(axis=0) / sw
        zz = (w[at, None, None] * (Z[at][:, :, None] * Z[at][:, None, :])
              ).sum(axis=0) / sw
        info += zz - np.outer(zbar, zbar)
        score_i[i] += Z[i] - zbar
    var = np.linalg.pinv(info)
    out = {"information": info, "variance": var,
           "se": np.sqrt(np.maximum(np.diag(var), 0.0))}
    if robust:
        if cluster is None:
            meat = score_i.T @ score_i
        else:
            cl = np.asarray(cluster).ravel()
            if cl.size != n:
                raise ValueError(f"cluster has {cl.size} entries for {n}.")
            agg = np.array([score_i[cl == l].sum(axis=0) for l in np.unique(cl)])
            meat = agg.T @ agg
        rob = var @ meat @ var
        out["robust_variance"] = rob
        out["robust_se"] = np.sqrt(np.maximum(np.diag(rob), 0.0))
        out["ratio"] = float(out["robust_se"][0] / out["se"][0]) \
            if out["se"][0] > 0 else np.nan
    else:
        out["robust_variance"] = None
        out["robust_se"] = None
        out["ratio"] = None
    out.update({
        "partial_likelihood_note": "the baseline hazard is profiled out, yet "
                                   "the observed information behaves as a real one",
        "diagnostic": "model-based and robust SEs diverge exactly when "
                      "proportional hazards fails",
        "n_events": int(ev.sum()), "n": int(n),
        "method": "Cox variance from the observed partial-likelihood information"})
    return RichResult(payload=out)


def cheatsheet():
    return "survvar: robust vs model-based SE divergence IS a proportional-hazards diagnostic"
