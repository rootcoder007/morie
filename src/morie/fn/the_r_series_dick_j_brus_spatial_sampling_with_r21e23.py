"""Gaussian log-likelihood of the geostatistical model.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_23"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_23(z, mu, cov):
    """Gaussian log-likelihood of the geostatistical model

    Formula: ln f(z|mu, theta) = -0.5(n ln 2pi + ln|C| + (z-mu)^T C^-1 (z-mu))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.23).
    """
    value = _brus.gaussian_loglikelihood(z, mu, cov)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.23)"
    return RichResult(
        title='Gaussian log-likelihood of the geostatistical model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e23: ln f(z|mu, theta) = -0.5(n ln 2pi + ln|C| + (z-mu)^T C^-1 (z-mu)) [Brus 2022, eq. 21.23]'
