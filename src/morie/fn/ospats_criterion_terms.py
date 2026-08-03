"""Stratification objective O = (sum w_h S_h)^2.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ospats_criterion_terms"]


def ospats_criterion_terms(w_h, s_h):
    """Stratification objective O = (sum w_h S_h)^2

    Formula: O = (sum_h w_h S_h)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (13.12).
    """
    value = _brus.ospats_criterion_terms(w_h, s_h)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (13.12)"
    return RichResult(
        title='Stratification objective O = (sum w_h S_h)^2',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r13e12: O = (sum_h w_h S_h)^2 [Brus 2022, eq. 13.12]'
