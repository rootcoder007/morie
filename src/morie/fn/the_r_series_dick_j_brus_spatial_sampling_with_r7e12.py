"""True variance of the pps two-stage total.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_12"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_12(p, t_j, t_total, m_j, f2_j, s2_j, m_j_sampled, n):
    """True variance of the pps two-stage total

    Formula: V(t_hat) = (1/n) sum p_j (t_j/p_j - t)^2 + (1/n) sum M_j^2 (1-f_2j) S2_j/(m_j p_j)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.12).
    """
    value = _brus.twostage_total_variance_pps(p, t_j, t_total, m_j, f2_j, s2_j, m_j_sampled, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.12)"
    return RichResult(
        title='True variance of the pps two-stage total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e12: V(t_hat) = (1/n) sum p_j (t_j/p_j - t)^2 + (1/n) sum M_j^2 (1-f_2j) S2_j/(m_j p_j) [Brus 2022, eq. 7.12]'
