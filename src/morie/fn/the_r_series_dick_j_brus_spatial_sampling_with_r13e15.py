"""Ospats expected squared discrepancy of two units.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15(zhat_i, zhat_j, r2, s2_i, s2_j, s2_ij):
    """Ospats expected squared discrepancy of two units

    Formula: E_xi[d2_ij] = (zhat_i - zhat_j)^2/R^2 + S2_i + S2_j - 2 S2_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.15).
    """
    value = _brus.expected_squared_distance(zhat_i, zhat_j, r2, s2_i, s2_j, s2_ij)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.15)"
    return RichResult(
        title='Ospats expected squared discrepancy of two units',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e15: E_xi[d2_ij] = (zhat_i - zhat_j)^2/R^2 + S2_i + S2_j - 2 S2_ij [Brus 2022, eq. 13.15]'
