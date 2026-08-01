"""Mixed-model calibration under SI.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_40"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_40(z_sample, b_si, m_all_mean, m_sample_mean):
    """Mixed-model calibration under SI

    Formula: zbar_MC = zbar_S + b_SI (mbar_pop - mbar_S)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.40).
    """
    value = _brus.mixed_calibration_si(z_sample, b_si, m_all_mean, m_sample_mean)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.40)"
    return RichResult(
        title='Mixed-model calibration under SI',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e40: zbar_MC = zbar_S + b_SI (mbar_pop - mbar_S) [Brus 2022, eq. 10.40]'
