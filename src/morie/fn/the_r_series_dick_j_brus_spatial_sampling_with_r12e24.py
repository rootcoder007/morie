"""Beta posterior density for a proportion.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_24"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_24(p, z, n, c, d):
    """Beta posterior density for a proportion

    Formula: f(p|z,n,c,d) = p^(z+c-1)(1-p)^(n-z+d-1)/B(z+c, n-z+d)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (12.24).
    """
    value = _brus.beta_posterior_pdf(p, z, n, c, d)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (12.24)"
    return RichResult(
        title='Beta posterior density for a proportion',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r12e24: f(p|z,n,c,d) = p^(z+c-1)(1-p)^(n-z+d-1)/B(z+c, n-z+d) [Brus 2022, eq. 12.24]'
