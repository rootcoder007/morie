"""GLS estimator across repeated surveys.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_15_equation_10(x, c, zhat):
    """GLS estimator across repeated surveys

    Formula: zhat_GLS = (X^T C^-1 X)^-1 X^T C^-1 zhat

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (15.10).
    """
    arr = np.asarray(_brus.gls_estimator(x, c, zhat), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (15.10)"
    return RichResult(
        title='GLS estimator across repeated surveys',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r15e10: zhat_GLS = (X^T C^-1 X)^-1 X^T C^-1 zhat [Brus 2022, eq. 15.10]'
