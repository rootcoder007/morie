"""Discrete average length criterion.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_25(lengths, probs, l_max):
    """Discrete average length criterion

    Formula: sum_z l(z,n) f(z,n) <= l_max

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'expected_length' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.25).
    """
    payload = dict(_brus.average_length_criterion(lengths, probs, l_max))
    value = payload['expected_length']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.25)"
    return RichResult(
        title='Discrete average length criterion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e25: sum_z l(z,n) f(z,n) <= l_max [Brus 2022, eq. 12.25]'
