"""Horvitz-Thompson estimator of the population total.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ht_total"]


def ht_total(z, pi):
    """Horvitz-Thompson estimator of the population total

    Formula: t_hat_pi(z) = sum_{k in S} w_k z_k with w_k = 1/pi_k

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (2.2).
    """
    value = _brus.ht_total(z, pi)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (2.2)"
    return RichResult(
        title='Horvitz-Thompson estimator of the population total',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r2e2: t_hat_pi(z) = sum_{k in S} w_k z_k with w_k = 1/pi_k [Brus 2022, eq. 2.2]'
