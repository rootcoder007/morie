"""Variance of the mean under autocorrelation.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_26_equation_3(sigma2, n, rho_bar):
    """Variance of the mean under autocorrelation

    Formula: V(mu_hat) = (sigma2/n)(1 + (n-1) rhobar)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (26.3).
    """
    value = _brus.autocorrelated_mean_variance(sigma2, n, rho_bar)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (26.3)"
    return RichResult(
        title='Variance of the mean under autocorrelation',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r26e3: V(mu_hat) = (sigma2/n)(1 + (n-1) rhobar) [Brus 2022, eq. 26.3]'
