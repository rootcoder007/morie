"""g-weighted variance estimator.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_18(g, e, n, n_population):
    """g-weighted variance estimator

    Formula: V_hat = (1 - n/N) sum g^2 e^2/(n(n-1))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.18).
    """
    value = _brus.g_weighted_variance(g, e, n, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.18)"
    return RichResult(
        title='g-weighted variance estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e18: V_hat = (1 - n/N) sum g^2 e^2/(n(n-1)) [Brus 2022, eq. 10.18]'
