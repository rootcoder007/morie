"""Ordinary kriging variance, semivariance form.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_11"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_11(lam, gamma_s0, nu):
    """Ordinary kriging variance, semivariance form

    Formula: V_OK = lam^T gamma0 + nu

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.11).
    """
    value = _brus.ok_variance_semivariance_form(lam, gamma_s0, nu)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.11)"
    return RichResult(
        title='Ordinary kriging variance, semivariance form',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e11: V_OK = lam^T gamma0 + nu [Brus 2022, eq. 21.11]'
