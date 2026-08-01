"""Variance of the calibration estimator via residuals.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_42"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_42(e, pi, n_population):
    """Variance of the calibration estimator via residuals

    Formula: V_hat(zbar_MC) = V_hat(ebar_pi)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.42).
    """
    value = _brus.mc_variance_via_residuals(e, pi, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.42)"
    return RichResult(
        title='Variance of the calibration estimator via residuals',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e42: V_hat(zbar_MC) = V_hat(ebar_pi) [Brus 2022, eq. 10.42]'
