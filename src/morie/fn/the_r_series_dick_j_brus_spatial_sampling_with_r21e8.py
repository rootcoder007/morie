"""Ordinary kriging variance, covariance form.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_8(sigma2, lam, cov_s0, nu):
    """Ordinary kriging variance, covariance form

    Formula: V_OK = sigma2 - lam^T c0 - nu

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.8).
    """
    value = _brus.ok_variance_covariance_form(sigma2, lam, cov_s0, nu)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.8)"
    return RichResult(
        title='Ordinary kriging variance, covariance form',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e8: V_OK = sigma2 - lam^T c0 - nu [Brus 2022, eq. 21.8]'
