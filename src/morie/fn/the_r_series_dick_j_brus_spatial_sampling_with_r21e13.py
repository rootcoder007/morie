"""Exponential semivariogram.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_13"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_13(h, c0, c1, phi):
    """Exponential semivariogram

    Formula: gamma(h) = 0 at h = 0, else c0 + c1(1 - exp(-h/phi)) (book prose: 95 percent of sill at 3 phi; the printed exp form is a display typo)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.13).
    """
    value = _brus.exponential_semivariogram(h, c0, c1, phi)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.13)"
    return RichResult(
        title='Exponential semivariogram',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e13: gamma(h) = 0 at h = 0, else c0 + c1(1 - exp(-h/phi)) (book prose: 95 percent of sill at 3 phi; the printed exp form is a display typo) [Brus 2022, eq. 21.13]'
