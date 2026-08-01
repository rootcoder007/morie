"""Sample proportion from 0/1 indicators.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_6"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_6(y):
    """Sample proportion from 0/1 indicators

    Formula: p_hat = (1/n) sum y_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (3.6).
    """
    value = _brus.si_proportion(y)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (3.6)"
    return RichResult(
        title='Sample proportion from 0/1 indicators',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r3e6: p_hat = (1/n) sum y_k [Brus 2022, eq. 3.6]'
