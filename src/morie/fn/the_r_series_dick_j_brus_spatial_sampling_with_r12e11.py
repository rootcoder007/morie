"""Required n for an interval length on a proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_11"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_11(u_crit, p_star, l_max):
    """Required n for an interval length on a proportion

    Formula: n = (u sqrt(p*(1-p*))/(l_max/2))^2 + 1

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.11).
    """
    value = _brus.n_for_proportion_length(u_crit, p_star, l_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.11)"
    return RichResult(
        title='Required n for an interval length on a proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e11: n = (u sqrt(p*(1-p*))/(l_max/2))^2 + 1 [Brus 2022, eq. 12.11]'
