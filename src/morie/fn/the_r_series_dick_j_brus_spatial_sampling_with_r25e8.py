"""Map-unit classification agreement indicator.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_25_equation_8(c_hat, c_true, u):
    """Map-unit classification agreement indicator

    Formula: y_k = 1 if chat_k = c_k = u else 0

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (25.8).
    """
    value = _brus.classification_indicator(c_hat, c_true, u)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (25.8)"
    return RichResult(
        title='Map-unit classification agreement indicator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r25e8: y_k = 1 if chat_k = c_k = u else 0 [Brus 2022, eq. 25.8]'
