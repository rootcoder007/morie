"""Variance estimator of a proportion under SI.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["si_proportion_variance"]


def si_proportion_variance(p_hat, n, n_population):
    """Variance estimator of a proportion under SI

    Formula: V_hat(p_hat) = (1 - n/N) p_hat(1-p_hat)/(n-1)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (3.14).
    """
    value = _brus.si_proportion_variance(p_hat, n, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (3.14)"
    return RichResult(
        title='Variance estimator of a proportion under SI',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r3e14: V_hat(p_hat) = (1 - n/N) p_hat(1-p_hat)/(n-1) [Brus 2022, eq. 3.14]'
