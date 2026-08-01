"""Regression estimator in slope form (multiple covariates).

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_9(zbar_pi, b_hats, xbar_true, xbar_pi):
    """Regression estimator in slope form (multiple covariates)

    Formula: zbar_regr = zbar_pi + sum b_j (xbar_j - xbar_hat_j)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.9).
    """
    value = _brus.regression_estimator_slopes(zbar_pi, b_hats, xbar_true, xbar_pi)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.9)"
    return RichResult(
        title='Regression estimator in slope form (multiple covariates)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e9: zbar_regr = zbar_pi + sum b_j (xbar_j - xbar_hat_j) [Brus 2022, eq. 10.9]'
