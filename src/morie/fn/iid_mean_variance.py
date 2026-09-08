"""Variance of the mean under iid sampling.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["iid_mean_variance"]


def iid_mean_variance(sigma2, n):
    """Variance of the mean under iid sampling

    Formula: V(mu_hat) = sigma2/n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (26.2).
    """
    value = _brus.iid_mean_variance(sigma2, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (26.2)"
    return RichResult(
        title='Variance of the mean under iid sampling',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r26e2: V(mu_hat) = sigma2/n [Brus 2022, eq. 26.2]'
