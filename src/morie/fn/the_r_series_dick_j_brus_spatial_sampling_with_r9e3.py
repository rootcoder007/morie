"""Variance estimator for balanced sampling.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_3(e, pi, c, n_population, p):
    """Variance estimator for balanced sampling

    Formula: V_hat = (1/N^2)(n/(n-p)) sum c_k (e_k/pi_k)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (9.3).
    """
    value = _brus.balanced_variance(e, pi, c, n_population, p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (9.3)"
    return RichResult(
        title='Variance estimator for balanced sampling',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r9e3: V_hat = (1/N^2)(n/(n-p)) sum c_k (e_k/pi_k)^2 [Brus 2022, eq. 9.3]'
