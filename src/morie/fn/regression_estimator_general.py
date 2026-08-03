"""Generalized regression estimator of the mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["regression_estimator_general"]


def regression_estimator_general(x_all, b_hat, z_sample, x_sample, pi_sample, n_population):
    """Generalized regression estimator of the mean

    Formula: zbar_regr = mean(x^T b_hat) + HT mean of residuals

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.8).
    """
    value = _brus.regression_estimator_general(x_all, b_hat, z_sample, x_sample, pi_sample, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.8)"
    return RichResult(
        title='Generalized regression estimator of the mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e8: zbar_regr = mean(x^T b_hat) + HT mean of residuals [Brus 2022, eq. 10.8]'
