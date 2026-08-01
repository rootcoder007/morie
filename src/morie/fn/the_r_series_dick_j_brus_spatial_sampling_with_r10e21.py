"""Combined regression estimator over strata.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21(zbar_pi, b_hat, xbar_true, xbar_pi):
    """Combined regression estimator over strata

    Formula: zbar_cregr = zbar_pi + b_hat (xbar - xbar_hat_pi)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.21).
    """
    value = _brus.regression_estimator_slopes(zbar_pi, [b_hat], [xbar_true], [xbar_pi])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.21)"
    return RichResult(
        title='Combined regression estimator over strata',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e21: zbar_cregr = zbar_pi + b_hat (xbar - xbar_hat_pi) [Brus 2022, eq. 10.21]'
