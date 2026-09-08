"""Equal-area geostrata simplification.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["mean_semivariance_equal_area"]


def mean_semivariance_equal_area(gamma_bar_h, n):
    """Equal-area geostrata simplification

    Formula: E_xi{V_STSI} = (1/n^2) sum gammabar_h

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.7).
    """
    value = _brus.mean_semivariance_equal_area(gamma_bar_h, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.7)"
    return RichResult(
        title='Equal-area geostrata simplification',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e7: E_xi{V_STSI} = (1/n^2) sum gammabar_h [Brus 2022, eq. 13.7]'
