"""Confidence interval estimate +/- u sqrt(V).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["confidence_interval"]


def confidence_interval(estimate, variance, u_crit):
    """Confidence interval estimate +/- u sqrt(V)

    Formula: zbar_hat -/+ u_(alpha/2) sqrt(V(zbar_hat))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'lower' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (3.15).
    """
    payload = dict(_brus.confidence_interval(estimate, variance, u_crit))
    value = payload['lower']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (3.15)"
    return RichResult(
        title='Confidence interval estimate +/- u sqrt(V)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r3e15: zbar_hat -/+ u_(alpha/2) sqrt(V(zbar_hat)) [Brus 2022, eq. 3.15]'
