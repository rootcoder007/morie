"""Local-mean variance estimator for well-spread samples.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["local_mean_variance"]


def local_mean_variance(e, pi, e_local_mean, n, p):
    """Local-mean variance estimator for well-spread samples

    Formula: V_hat = (n/(n-p))(p/(p+1)) sum (1-pi)(e/pi - ebar_local)^2

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (9.10).
    """
    value = _brus.local_mean_variance(e, pi, e_local_mean, n, p)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (9.10)"
    return RichResult(
        title='Local-mean variance estimator for well-spread samples',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r9e10: V_hat = (n/(n-p))(p/(p+1)) sum (1-pi)(e/pi - ebar_local)^2 [Brus 2022, eq. 9.10]'
