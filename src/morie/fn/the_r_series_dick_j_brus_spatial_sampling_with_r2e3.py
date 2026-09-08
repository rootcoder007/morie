"""Design weight w_k = 1 / pi_k.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_2_equation_3(pi_k):
    """Design weight w_k = 1 / pi_k

    Formula: w_k = 1 / pi_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (2.3).
    """
    value = 1.0 / float(pi_k) if 0 < float(pi_k) <= 1 else (_ for _ in ()).throw(ValueError('need 0 < pi <= 1'))
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (2.3)"
    return RichResult(
        title='Design weight w_k = 1 / pi_k',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r2e3: w_k = 1 / pi_k [Brus 2022, eq. 2.3]'
