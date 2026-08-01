"""Simple linear working model for mapping.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_16_equation_1"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_16_equation_1(beta0, beta1, x):
    """Simple linear working model for mapping

    Formula: Z_k = beta0 + beta1 x_k + eps_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (16.1).
    """
    value = _brus.linear_model_prediction(beta0, beta1, x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (16.1)"
    return RichResult(
        title='Simple linear working model for mapping',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r16e1: Z_k = beta0 + beta1 x_k + eps_k [Brus 2022, eq. 16.1]'
