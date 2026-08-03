"""Variance estimator of the ratio total.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ratio_total_variance"]


def ratio_total_variance(e, n, n_population):
    """Variance estimator of the ratio total

    Formula: V_hat(t_ratio) = N^2 S2_hat(e)/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.25).
    """
    value = _brus.ratio_total_variance(e, n, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.25)"
    return RichResult(
        title='Variance estimator of the ratio total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e25: V_hat(t_ratio) = N^2 S2_hat(e)/n [Brus 2022, eq. 10.25]'
