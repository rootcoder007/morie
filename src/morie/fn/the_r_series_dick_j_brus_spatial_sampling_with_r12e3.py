"""Required n for a standard error target on a proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_3(p_star, se_max):
    """Required n for a standard error target on a proportion

    Formula: n = (sqrt(p*(1-p*))/se_max)^2 + 1

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.3).
    """
    value = _brus.n_for_proportion_se(p_star, se_max)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.3)"
    return RichResult(
        title='Required n for a standard error target on a proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e3: n = (sqrt(p*(1-p*))/se_max)^2 + 1 [Brus 2022, eq. 12.3]'
