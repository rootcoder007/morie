"""Optimal number of SSUs per PSU.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_10(s_w, s_b, c1, c2):
    """Optimal number of SSUs per PSU

    Formula: m = (S_w/S_b) sqrt(c1/c2)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.10).
    """
    value = _brus.twostage_optimal_m(s_w, s_b, c1, c2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.10)"
    return RichResult(
        title='Optimal number of SSUs per PSU',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e10: m = (S_w/S_b) sqrt(c1/c2) [Brus 2022, eq. 7.10]'
