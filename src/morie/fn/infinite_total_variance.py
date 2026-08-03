"""Variance of the infinite-population total estimator.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["infinite_total_variance"]


def infinite_total_variance(s2_hat, n, area, sample_area):
    """Variance of the infinite-population total estimator

    Formula: V_hat(t_hat) = (A/a)^2 S2_hat(z)/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (3.21).
    """
    value = _brus.infinite_total_variance(s2_hat, n, area, sample_area)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (3.21)"
    return RichResult(
        title='Variance of the infinite-population total estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r3e21: V_hat(t_hat) = (A/a)^2 S2_hat(z)/n [Brus 2022, eq. 3.21]'
