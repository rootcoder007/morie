"""Universal kriging with a polynomial trend."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["universal_kriging"]


def universal_kriging(coords, values, s_predict, trend_order=1,
                      model="exponential", nugget=0.0, sill=1.0, range_=1.0):
    r"""Universal kriging: ordinary kriging of the residuals from a polynomial trend.

    .. math::
        Z(s) = \mu(s) + \delta(s), \qquad
        \mu(s) = \sum_k \beta_k f_k(s),

    with :math:`\delta` zero mean and second-order stationary. The
    predictor solves the augmented system that enforces unbiasedness on
    every trend basis function, so the trend coefficients never have to
    be estimated separately.

    The previous body was a placeholder: it averaged ``coords`` and used
    neither ``values``, ``s_predict`` nor ``trend_order``.

    Parameters
    ----------
    coords : array-like
        Data locations, shape ``(n, d)``.
    values : array-like
        Observations, length ``n``.
    s_predict : array-like
        Target locations, shape ``(m, d)`` or ``(d,)``.
    trend_order : {0, 1, 2}, default 1
        Polynomial order of the mean. Order 0 is ordinary kriging.
    model : {'exponential', 'gaussian', 'spherical'}
        Covariance model.
    nugget, sill, range_ : float
        :math:`c_0`, the total sill, and the range.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``n``, ``trend_order``, ``method``.

    Notes
    -----
    Order 0 makes the trend basis a single column of ones, which is
    exactly the ordinary kriging constraint :math:`\sum_i\lambda_i = 1`;
    :func:`morie.fn.krigFDA.kriging` returns the same predictions from
    the explicit GLS form.

    References
    ----------
    Cressie, N. A. C. (1993). *Statistics for Spatial Data*, rev. edn.
    Wiley, sec. 3.4.5.

    Schabenberger, O. & Gotway, C. A. (2005). *Statistical Methods for
    Spatial Data Analysis*. Chapman & Hall/CRC, ch. 5.
    """
    from .ukrig import universal_kriging as _uk

    res = _uk(values, coords, s_predict, model, nugget, sill, range_,
              trend_order)
    est = res["estimate"]
    se = res["se"]
    if not isinstance(est, list):
        est = [est]
        se = [se]
    return RichResult(
        payload={
            "estimate": [float(v) for v in est],
            "se": [float(v) for v in se],
            "n": int(res["n"]),
            "trend_order": int(trend_order),
            "method": "Universal kriging with a polynomial trend of order %d"
                      % (int(trend_order),),
        }
    )


def cheatsheet():
    return "krigun: universal kriging, polynomial trend of order 0, 1 or 2"
