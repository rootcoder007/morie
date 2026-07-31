"""Leave-one-out cross-validation for kriging: MSPE."""

import numpy as np

from ._richresult import RichResult
from ._schab_krig import simple_kriging

__all__ = ["schabenberger_cross_validation_kriging"]


def schabenberger_cross_validation_kriging(coords, z, cov_model=None, mu=None):
    r"""
    Leave-one-out cross-validation of a kriging model.

    Kriging honours the data, so in-sample residuals are identically
    zero and carry no information about model fit. Cross-validation
    removes each observation in turn, predicts it from the rest, and
    reports

    .. math::

        \mathrm{MSPE} = \frac{1}{n}\sum_i (Z(s_i) - \hat{Z}_{-i}(s_i))^2

    The standardised residuals
    :math:`(Z(s_i) - \hat{Z}_{-i}) / \sigma_{-i}` should have mean near
    zero and variance near one if the covariance model is right; their
    variance is the diagnostic that catches a mis-specified sill.

    Parameters
    ----------
    coords : array-like
        Observation coordinates, shape ``(n, d)``.
    z : array-like
        Observed values, shape ``(n,)``.
    cov_model : mapping, optional
        ``{'model', 'nugget', 'sill', 'range'}``.
    mu : float, optional
        Known mean; the sample mean of the retained points when omitted.

    Returns
    -------
    RichResult
        ``mspe``, ``rmspe``, ``me`` (mean error), ``residuals``,
        ``standardised``, ``std_variance``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Ch. 5.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    z = np.asarray(z, dtype=float).ravel()
    n = z.size
    if coords.shape[0] != n:
        raise ValueError("`coords` and `z` must have the same number of rows")
    if n < 3:
        raise ValueError("leave-one-out cross-validation needs at least 3 points")
    resid = np.empty(n)
    sd = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        p, v, _ = simple_kriging(coords[keep], z[keep], coords[i:i + 1],
                                 cov_model, mu)
        resid[i] = z[i] - float(p[0])
        sd[i] = np.sqrt(max(float(v[0]), 1e-300))
    std = resid / sd
    return RichResult(
        title="Leave-one-out cross-validation of kriging",
        summary_lines=[("MSPE", float(np.mean(resid**2))),
                       ("mean error", float(np.mean(resid))),
                       ("var of standardised residuals", float(np.var(std)))],
        payload={"mspe": float(np.mean(resid**2)),
                 "rmspe": float(np.sqrt(np.mean(resid**2))),
                 "me": float(np.mean(resid)), "residuals": resid,
                 "standardised": std, "std_variance": float(np.var(std)), "n": n},
    )


def cheatsheet():
    return "spkfnn: leave-one-out kriging CV; MSPE and standardised residuals."
