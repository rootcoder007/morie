"""Required n for a relative error target.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_10(u_crit, cv_star, r_max):
    """Required n for a relative error target

    Formula: n = (u cv*/r_max)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.10).
    """
    value = _brus.n_for_cv(u_crit, cv_star, r_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.10)"
    return RichResult(
        title='Required n for a relative error target',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e10: n = (u cv*/r_max)^2 [Brus 2022, eq. 12.10]'
