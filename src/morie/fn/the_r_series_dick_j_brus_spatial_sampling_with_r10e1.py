"""Model-assisted working model Z_k = m(x_k) + eps_k.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_1"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_1(m_x, eps):
    """Model-assisted working model Z_k = m(x_k) + eps_k

    Formula: Z_k = mu(x_k) + eps_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.1).
    """
    value = float(m_x) + float(eps)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.1)"
    return RichResult(
        title='Model-assisted working model Z_k = m(x_k) + eps_k',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e1: Z_k = mu(x_k) + eps_k [Brus 2022, eq. 10.1]'
