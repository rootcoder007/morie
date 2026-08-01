"""Two-stage variance estimator from PSU means.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_7(primary_unit_means):
    """Two-stage variance estimator from PSU means

    Formula: V_hat = S2_hat(zbar)/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'variance' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.7).
    """
    payload = dict(_brus.twostage_variance_estimator(primary_unit_means))
    value = payload['variance']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.7)"
    return RichResult(
        title='Two-stage variance estimator from PSU means',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e7: V_hat = S2_hat(zbar)/n [Brus 2022, eq. 7.7]'
