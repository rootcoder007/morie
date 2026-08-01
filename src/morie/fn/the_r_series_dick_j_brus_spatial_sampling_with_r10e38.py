"""Calibration intercept a_hat = (1 - b_hat) HT mean of z.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_38"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_38(b_hat, z_sample, pi_sample, n_population):
    """Calibration intercept a_hat = (1 - b_hat) HT mean of z

    Formula: a_hat = (1 - b_hat)(1/N) sum z_k/pi_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.38).
    """
    value = _brus.mixed_calibration_intercept(b_hat, z_sample, pi_sample, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.38)"
    return RichResult(
        title='Calibration intercept a_hat = (1 - b_hat) HT mean of z',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e38: a_hat = (1 - b_hat)(1/N) sum z_k/pi_k [Brus 2022, eq. 10.38]'
