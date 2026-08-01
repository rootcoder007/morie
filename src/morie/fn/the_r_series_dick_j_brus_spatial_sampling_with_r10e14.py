"""Residual variance S2_hat(e) = sum e^2/(n-1).

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_14(e, n, n_population):
    """Residual variance S2_hat(e) = sum e^2/(n-1)

    Formula: S2_hat(e) = (1/(n-1)) sum e_k^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 's2_e' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.14).
    """
    payload = dict(_brus.si_regression_variance(e, n, n_population))
    value = payload['s2_e']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.14)"
    return RichResult(
        title='Residual variance S2_hat(e) = sum e^2/(n-1)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e14: S2_hat(e) = (1/(n-1)) sum e_k^2 [Brus 2022, eq. 10.14]'
