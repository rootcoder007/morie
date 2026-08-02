"""Estimation-adjusted criterion EAC.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_6"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_24_equation_6(akv, v_ok, vkv):
    """Estimation-adjusted criterion EAC

    Formula: EAC = AKV + VKV/(2 V_OK)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.6).
    """
    value = _brus.estimation_adjusted_criterion(akv, v_ok, vkv)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.6)"
    return RichResult(
        title='Estimation-adjusted criterion EAC',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e6: EAC = AKV + VKV/(2 V_OK) [Brus 2022, eq. 24.6]'
