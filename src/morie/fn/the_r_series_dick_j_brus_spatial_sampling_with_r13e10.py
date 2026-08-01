"""Variance under optimal allocation with costs.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_10(weights, s_h, c_h, n):
    """Variance under optimal allocation with costs

    Formula: V = (1/n)(sum w S sqrt(c))(sum w S/sqrt(c))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.10).
    """
    value = _brus.optimal_allocation_variance(weights, s_h, c_h, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.10)"
    return RichResult(
        title='Variance under optimal allocation with costs',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e10: V = (1/n)(sum w S sqrt(c))(sum w S/sqrt(c)) [Brus 2022, eq. 13.10]'
