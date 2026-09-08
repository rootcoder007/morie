"""Variance of the kriging variance (VKV).

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["variance_of_kriging_variance"]


def variance_of_kriging_variance(cov_theta, dv_dtheta):
    """Variance of the kriging variance (VKV)

    Formula: VKV = sum_ij Cov(th_i, th_j) dV/dth_i dV/dth_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (24.3).
    """
    value = _brus.variance_of_kriging_variance(cov_theta, dv_dtheta)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (24.3)"
    return RichResult(
        title='Variance of the kriging variance (VKV)',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r24e3: VKV = sum_ij Cov(th_i, th_j) dV/dth_i dV/dth_j [Brus 2022, eq. 24.3]'
