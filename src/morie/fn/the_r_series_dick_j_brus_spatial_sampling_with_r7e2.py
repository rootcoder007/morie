"""Two-stage mean of primary-unit means.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_2(primary_unit_means):
    """Two-stage mean of primary-unit means

    Formula: zbarbar_hat = (1/n) sum zbar_hat_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (7.2).
    """
    value = _brus.twostage_mean(primary_unit_means)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (7.2)"
    return RichResult(
        title='Two-stage mean of primary-unit means',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r7e2: zbarbar_hat = (1/n) sum zbar_hat_j [Brus 2022, eq. 7.2]'
