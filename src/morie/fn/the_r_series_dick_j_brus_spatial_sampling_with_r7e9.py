"""Optimal number of PSUs for a variance target.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_9"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_9(s_w, s_b, c1, c2, v_max):
    """Optimal number of PSUs for a variance target

    Formula: n = (S_w S_b sqrt(c2/c1) + S_b^2)/V_max

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.9).
    """
    value = _brus.twostage_optimal_n_variance(s_w, s_b, c1, c2, v_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.9)"
    return RichResult(
        title='Optimal number of PSUs for a variance target',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e9: n = (S_w S_b sqrt(c2/c1) + S_b^2)/V_max [Brus 2022, eq. 7.9]'
