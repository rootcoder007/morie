"""g-weight of the simple regression estimator.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["g_weight_simple"]


def g_weight_simple(x_k, xbar_true, xbar_sample, s2_x):
    """g-weight of the simple regression estimator

    Formula: g_k = 1 + (xbar - xbar_S)(x_k - xbar_S)/S2_hat(x)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.17).
    """
    value = _brus.g_weight_simple(x_k, xbar_true, xbar_sample, s2_x)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.17)"
    return RichResult(
        title='g-weight of the simple regression estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e17: g_k = 1 + (xbar - xbar_S)(x_k - xbar_S)/S2_hat(x) [Brus 2022, eq. 10.17]'
