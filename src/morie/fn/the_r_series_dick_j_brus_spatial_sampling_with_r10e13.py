"""Variance of the SI regression estimator.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_13(e, n, n_population):
    """Variance of the SI regression estimator

    Formula: V_hat = (1 - n/N) S2_hat(e)/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'variance' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.13).
    """
    payload = dict(_brus.si_regression_variance(e, n, n_population))
    value = payload['variance']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.13)"
    return RichResult(
        title='Variance of the SI regression estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e13: V_hat = (1 - n/N) S2_hat(e)/n [Brus 2022, eq. 10.13]'
