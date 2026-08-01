"""Cluster-sampling mean zbarbar_hat = t_hat / M.

Book-as-spec implementation; see reference for context.
"""

import numpy as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_10"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_6_equation_10(t_hat, m_population):
    """Cluster-sampling mean zbarbar_hat = t_hat / M

    Formula: zbarbar_hat_pi = t_hat(z)/M

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (6.10).
    """
    value = _brus.cluster_mean_from_total(t_hat, m_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (6.10)"
    return RichResult(
        title='Cluster-sampling mean zbarbar_hat = t_hat / M',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r6e10: zbarbar_hat_pi = t_hat(z)/M [Brus 2022, eq. 6.10]'
