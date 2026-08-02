"""Difference estimator of the mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_2(m_all, z_sample, m_sample, pi_sample, n_population):
    """Difference estimator of the mean

    Formula: zbar_dif = mean of model predictions + HT mean of residuals

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.2).
    """
    value = _brus.difference_estimator(m_all, z_sample, m_sample, pi_sample, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.2)"
    return RichResult(
        title='Difference estimator of the mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e2: zbar_dif = mean of model predictions + HT mean of residuals [Brus 2022, eq. 10.2]'
