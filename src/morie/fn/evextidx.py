# morie.fn -- function file (rootcoder007/morie)
"""Runs estimator of the extremal index."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_extremal_runs", "evt_extremal_index_runs"]


def ev_extremal_runs(x, threshold, run_length=1):
    r"""The runs estimator of the extremal index :math:`\theta`
    (Smith and Weissman 1994):

    .. math:: \hat\theta_R = \frac{\#\{\text{clusters}\}}
                                  {\#\{\text{exceedances}\}},

    where a new cluster begins whenever an exceedance of the
    threshold is separated from the previous one by more than
    ``run_length`` non-exceedances.

    :math:`\theta \in (0, 1]` is the reciprocal MEAN CLUSTER SIZE of
    exceedances in a stationary series: independent data have
    :math:`\theta = 1`, and :math:`\theta = 1/2` means exceedances
    arrive in clumps of two on average -- so the "effective number"
    of independent extremes is :math:`\theta n`, and ignoring
    :math:`\theta` overstates every return level. For the max-AR
    process :math:`X_t = \max(\alpha X_{t-1}, (1-\alpha)Z_t)` the
    index is exactly :math:`1 - \alpha`, which is what the tests
    use as an oracle.

    The estimator's known weakness is its run-length sensitivity:
    too short splits genuine clusters, too long merges distinct
    ones, and there is no data-free right answer. The
    intervals estimator (``morie.fn.evextint``) removes that tuning
    parameter and is the usual cross-check.

    Parameters
    ----------
    x : array-like
        Stationary series.
    threshold : float
        Exceedance threshold; a high quantile of the series.
    run_length : int, default 1
        Separation (in non-exceedances) that starts a new cluster.

    Returns
    -------
    RichResult
        keys: ``theta``, ``n_exceedances``, ``n_clusters``,
        ``mean_cluster_size``, ``run_length``, ``threshold``,
        ``interpretation``, ``sensitivity_note``, ``n``, ``method``.

    References
    ----------
    Smith, R. L. and Weissman, I. (1994), "Estimating the extremal
    index", *JRSS-B* 56:515-528.
    """
    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 20:
        raise ValueError(f"need at least 20 observations, got {n}.")
    u = float(threshold)
    r = int(run_length)
    if r < 1:
        raise ValueError(f"run_length must be at least 1, got {r}.")
    exc = np.flatnonzero(xv > u)
    ne = exc.size
    if ne < 2:
        raise ValueError(
            f"only {ne} exceedance(s) of {u}; lower the threshold.")
    gaps = np.diff(exc)
    nc = 1 + int(np.sum(gaps > r))
    theta = nc / ne
    return RichResult(payload={
        "theta": float(theta),
        "n_exceedances": int(ne), "n_clusters": int(nc),
        "mean_cluster_size": float(ne / nc),
        "run_length": r, "threshold": u,
        "interpretation": "theta is the reciprocal mean cluster size: the "
                          "effective number of independent extremes is "
                          "theta * n, and ignoring it overstates every "
                          "return level",
        "sensitivity_note": "run_length too short splits genuine clusters, "
                            "too long merges distinct ones; the intervals "
                            "estimator (evextint) has no such tuning "
                            "parameter and is the usual cross-check",
        "n": int(n),
        "method": "Runs estimator of the extremal index (Smith-Weissman 1994)"})


def cheatsheet():
    return "evextidx: clusters / exceedances -- and the run length is a real tuning parameter"


#: Catalogue alias for :func:`ev_extremal_runs`.
evt_extremal_index_runs = ev_extremal_runs
