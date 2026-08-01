"""Variance of the stratified estimator.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_4_equation_4(stratum_variances, stratum_weights):
    """Variance of the stratified estimator

    Formula: V_hat = sum_h w_h^2 V_hat(zbar_hat_h)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (4.4).
    """
    value = _brus.stratified_variance(stratum_variances, stratum_weights)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (4.4)"
    return RichResult(
        title='Variance of the stratified estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r4e4: V_hat = sum_h w_h^2 V_hat(zbar_hat_h) [Brus 2022, eq. 4.4]'
