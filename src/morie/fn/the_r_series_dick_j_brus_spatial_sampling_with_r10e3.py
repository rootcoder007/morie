"""Linear working model Z_k = x_k^T beta + eps_k.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_3(x_k, beta, eps):
    """Linear working model Z_k = x_k^T beta + eps_k

    Formula: Z_k = x_k^T beta + eps_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.3).
    """
    value = float(np.dot(np.asarray(x_k, dtype=float), np.asarray(beta, dtype=float)) + eps)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.3)"
    return RichResult(
        title='Linear working model Z_k = x_k^T beta + eps_k',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e3: Z_k = x_k^T beta + eps_k [Brus 2022, eq. 10.3]'
