"""Constant-mean model Z(s) = mu + eps(s).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_21_equation_2(mu, eps):
    """Constant-mean model Z(s) = mu + eps(s)

    Formula: Z(s) = mu + eps(s)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (21.2).
    """
    value = float(mu) + float(eps)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (21.2)"
    return RichResult(
        title='Constant-mean model Z(s) = mu + eps(s)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r21e2: Z(s) = mu + eps(s) [Brus 2022, eq. 21.2]'
