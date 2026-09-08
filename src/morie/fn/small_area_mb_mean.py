"""Model-based small-area mean.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["small_area_mb_mean"]


def small_area_mb_mean(xbar_d, beta_hat, v_d):
    """Model-based small-area mean

    Formula: zbar_mb,d = xbar_d^T beta_hat + v_hat_d

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (14.15).
    """
    value = _brus.small_area_mb_mean(xbar_d, beta_hat, v_d)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (14.15)"
    return RichResult(
        title='Model-based small-area mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r14e15: zbar_mb,d = xbar_d^T beta_hat + v_hat_d [Brus 2022, eq. 14.15]'
