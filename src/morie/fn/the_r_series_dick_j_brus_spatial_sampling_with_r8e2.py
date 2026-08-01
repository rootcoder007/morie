"""pps with-replacement variance estimator of the total.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_8_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_8_equation_2(z, p, t_hat):
    """pps with-replacement variance estimator of the total

    Formula: V_hat(t_hat) = sum(z_k/p_k - t_hat)^2/(n(n-1))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (8.2).
    """
    value = _brus.pps_total_variance(z, p, t_hat)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (8.2)"
    return RichResult(
        title='pps with-replacement variance estimator of the total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r8e2: V_hat(t_hat) = sum(z_k/p_k - t_hat)^2/(n(n-1)) [Brus 2022, eq. 8.2]'
