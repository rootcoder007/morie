"""Augmented kriging variance AKV = V_OK + E[tau2].

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_4"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_4(v_ok, e_tau2):
    """Augmented kriging variance AKV = V_OK + E[tau2]

    Formula: AKV = V_OK + E[tau^2]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.4).
    """
    value = _brus.augmented_kriging_variance(v_ok, e_tau2)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.4)"
    return RichResult(
        title='Augmented kriging variance AKV = V_OK + E[tau2]',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e4: AKV = V_OK + E[tau^2] [Brus 2022, eq. 24.4]'
