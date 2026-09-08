# morie.fn -- function file (rootcoder007/morie)
"""Regression estimator (linear adjustment to known X-mean)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["regression_estimator"]


def regression_estimator(y, x, weights=None, X_mean=None):
    r"""Regression (difference) estimator:

    .. math:: \hat{\bar Y}_{reg} = \bar y
              + b\,(\bar X - \bar x),

    correcting the sample mean by the amount the auxiliary happens to
    be off in this sample.

    It dominates the ratio estimator in one specific way: the ratio
    estimator implicitly forces the fitted line through the ORIGIN,
    while this one fits an intercept. When the relationship between
    y and x does not pass through the origin the ratio estimator
    carries a bias the regression estimator does not, and the
    returned ``intercept`` shows how far from the origin the fit
    actually sits.

    Unlike the ratio estimator it is essentially unbiased to first
    order, and it never does worse than the simple mean
    asymptotically -- its variance is :math:`(1-\rho^2)` times the
    simple-mean variance, so the gain is exactly the squared
    correlation.

    Parameters
    ----------
    y, x : array-like
        Study and auxiliary variables.
    weights : array-like, optional
        Design weights.
    X_mean : float
        Known population mean of x. Required.

    Returns
    -------
    RichResult
        keys: ``mean``, ``slope``, ``intercept``, ``correlation``,
        ``variance_ratio_to_simple_mean``, ``passes_through_origin``,
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
    if X_mean is None:
        raise ValueError("the regression estimator needs the known population "
                         "mean of x.")
    w = np.ones(n) if weights is None else check_weights(weights, n)
    sw = float(w.sum())
    xbar = float(np.sum(w * xv) / sw)
    ybar = float(np.sum(w * yv) / sw)
    sxx = float(np.sum(w * (xv - xbar) ** 2))
    if sxx == 0:
        raise ValueError("the auxiliary has no weighted variation.")
    b = float(np.sum(w * (xv - xbar) * (yv - ybar)) / sxx)
    rho = float(np.corrcoef(xv, yv)[0, 1])
    return RichResult(payload={
        "mean": ybar + b * (float(X_mean) - xbar),
        "slope": b, "intercept": ybar - b * xbar,
        "correlation": rho,
        "variance_ratio_to_simple_mean": float(1.0 - rho ** 2),
        "passes_through_origin": bool(abs(ybar - b * xbar) <
                                      1e-8 * max(abs(ybar), 1.0)),
        "n": int(n),
        "method": "Regression estimator; fits an intercept, so no origin assumption, variance (1 - rho^2) times simple"})


def cheatsheet():
    return "regest: the ratio estimator forces the line through the origin -- this one does not"
