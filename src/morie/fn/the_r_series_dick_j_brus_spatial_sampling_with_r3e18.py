"""Total of an infinite population t_hat = (A/a) zbar_hat.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_3_equation_18(zbar_hat, area, sample_area):
    """Total of an infinite population t_hat = (A/a) zbar_hat

    Formula: t_hat(z) = (A/a) zbar_hat

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (3.18).
    """
    value = _brus.infinite_total(zbar_hat, area, sample_area)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (3.18)"
    return RichResult(
        title='Total of an infinite population t_hat = (A/a) zbar_hat',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r3e18: t_hat(z) = (A/a) zbar_hat [Brus 2022, eq. 3.18]'
