# morie.fn -- function file (rootcoder007/morie)
"""Ratio estimator (auxiliary X)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ratio_estimator"]


def ratio_estimator(y, x, weights=None, X_total=None, X_mean=None):
    r"""Ratio estimator with an auxiliary variable:

    .. math:: \hat R = \frac{\sum w_i y_i}{\sum w_i x_i},
              \qquad
              \hat{\bar Y}_R = \hat R \,\bar X .

    Borrows strength from a variable whose population total is
    KNOWN. It beats the plain mean when y is roughly proportional to
    x through the origin, and the condition is sharp: the ratio
    estimator improves on the simple mean when the correlation
    exceeds :math:`CV(x)/(2\,CV(y))`. That threshold is computed and
    returned rather than left as folklore, so a caller can see
    whether the auxiliary is actually earning its place.

    The estimator is biased, of order :math:`1/n`, for the same
    reason the Hajek estimator is.

    Parameters
    ----------
    y, x : array-like
        Study and auxiliary variables on the sample.
    weights : array-like, optional
        Design weights; equal weights otherwise.
    X_total, X_mean : float, optional
        The known population total or mean of x. One is required.

    Returns
    -------
    RichResult
        keys: ``ratio``, ``mean``, ``total``,
        ``improves_on_simple_mean``, ``efficiency_threshold``,
        ``correlation``, ``cv_x``, ``cv_y``, ``biased`` (True),
        ``n``, ``method``.
    """
    from ._survey import check_weights

    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).ravel()
    if xv.size != yv.size:
        raise ValueError(f"x has {xv.size} entries for {yv.size} of y.")
    n = yv.size
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}.")
    w = np.ones(n) if weights is None else check_weights(weights, n)
    if X_total is None and X_mean is None:
        raise ValueError("the ratio estimator needs the population total or "
                         "mean of x; that is what it borrows strength from.")
    denom = float(np.sum(w * xv))
    if denom == 0:
        raise ValueError("the weighted total of x is zero.")
    R = float(np.sum(w * yv) / denom)
    xbar_pop = float(X_mean) if X_mean is not None else \
        float(X_total) / float(np.sum(w))
    cvx = float(np.std(xv, ddof=1) / np.mean(xv)) if np.mean(xv) != 0 else np.inf
    cvy = float(np.std(yv, ddof=1) / np.mean(yv)) if np.mean(yv) != 0 else np.inf
    rho = float(np.corrcoef(xv, yv)[0, 1])
    thr = cvx / (2.0 * cvy) if np.isfinite(cvy) and cvy != 0 else np.inf
    return RichResult(payload={
        "ratio": R, "mean": R * xbar_pop,
        "total": R * (float(X_total) if X_total is not None
                      else xbar_pop * float(np.sum(w))),
        "improves_on_simple_mean": bool(rho > thr),
        "efficiency_threshold": thr, "correlation": rho,
        "cv_x": cvx, "cv_y": cvy, "biased": True,
        "n": int(n),
        "method": "Ratio estimator; beats the simple mean when rho > CV(x)/(2 CV(y))"})


def cheatsheet():
    return "ratest: the auxiliary earns its place only when rho > CV(x)/(2 CV(y)) -- computed, not assumed"
